#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import subprocess
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from model_analyzer.reports.html_report import HTMLReport
from model_analyzer.reports.report_factory import ReportFactory

from .common import test_result_collector as trc

_BLOCKED_MODULES = ("weasyprint", "model_analyzer.reports.pdf_report")


def _weasyprint_importable():
    try:
        import weasyprint  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


@contextmanager
def weasyprint_broken(error):
    """Make importing weasyprint fail the way a real environment does.

    ``None`` in sys.modules raises ImportError on import -- the "pdf" extra
    is not installed. An object whose import raises OSError models the other
    real case: WeasyPrint present, its native libraries missing.
    """

    saved = {
        name: sys.modules.pop(name) for name in _BLOCKED_MODULES if name in sys.modules
    }
    if error is ImportError:
        sys.modules["weasyprint"] = None
    else:

        class _Exploding:
            def __getattr__(self, name):
                raise OSError("cannot load library 'libgobject-2.0-0'")

        sys.modules["weasyprint"] = _Exploding()
        sys.modules["weasyprint.urls"] = _Exploding()
    try:
        yield
    finally:
        sys.modules.pop("weasyprint", None)
        sys.modules.pop("weasyprint.urls", None)
        sys.modules.update(saved)


class TestReportFactory(trc.TestResultCollector):
    """
    Tests that report selection degrades to HTML rather than failing when the
    optional PDF stack is absent or unusable.
    """

    def setUp(self):
        ReportFactory.WARNING_PRINTED = False

    def tearDown(self):
        ReportFactory.WARNING_PRINTED = False
        patch.stopall()

    def test_model_analyzer_imports_without_weasyprint(self):
        """model_analyzer must stay importable without the optional extra.

        This is the regression that matters: a module-scope import of
        weasyprint anywhere in the import graph runs before the guard in
        create_report, so it would break every Model Analyzer command and
        not just reporting.

        Run in a subprocess rather than by reloading in place -- a reload
        rebinds the module's own ReportFactory while this module keeps the
        original class, and the two then disagree about WARNING_PRINTED.
        """

        script = (
            "import sys\n"
            "sys.modules['weasyprint'] = None\n"
            "import model_analyzer.entrypoint\n"
            "import model_analyzer.reports.report_factory\n"
            "import model_analyzer.reports.report_manager\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_html_is_used_when_weasyprint_is_not_installed(self):
        with weasyprint_broken(ImportError):
            report = ReportFactory.create_report()
        self.assertIsInstance(report, HTMLReport)
        self.assertNotEqual(type(report).__name__, "PDFReport")

    def test_html_is_used_when_weasyprints_native_libraries_are_missing(self):
        """
        WeasyPrint raises OSError, not ImportError, when Pango and friends
        are absent. Catching only ImportError would crash the run instead of
        falling back.
        """

        with weasyprint_broken(OSError):
            report = ReportFactory.create_report()
        self.assertIsInstance(report, HTMLReport)
        self.assertNotEqual(type(report).__name__, "PDFReport")

    def test_pdf_is_used_when_weasyprint_is_usable(self):
        if not _weasyprint_importable():
            self.skipTest("requires the optional weasyprint dependency")
        report = ReportFactory.create_report()
        self.assertEqual(type(report).__name__, "PDFReport")

    def test_the_fallback_warning_is_printed_once(self):
        with weasyprint_broken(ImportError):
            with self.assertLogs(level="WARNING") as captured:
                ReportFactory.create_report()
                ReportFactory.create_report()
        self.assertEqual(len(captured.records), 1)

    def test_the_fallback_warning_names_the_extra(self):
        with weasyprint_broken(ImportError):
            with self.assertLogs(level="WARNING") as captured:
                ReportFactory.create_report()
        self.assertIn("triton-model-analyzer[pdf]", captured.records[0].getMessage())


if __name__ == "__main__":
    unittest.main()
