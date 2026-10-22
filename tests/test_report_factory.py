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


@contextmanager
def pdfkit_unavailable():
    """Make ``import pdfkit`` fail, as it does without the "pdf" extra.

    A ``None`` entry in ``sys.modules`` is the documented way to make an
    import raise ``ImportError`` without touching the filesystem. Any module
    that already imported pdfkit is evicted so it re-imports under the block.
    """
    blocked = ["pdfkit", "model_analyzer.reports.pdf_report"]
    saved = {name: sys.modules.pop(name) for name in blocked if name in sys.modules}
    sys.modules["pdfkit"] = None
    try:
        yield
    finally:
        del sys.modules["pdfkit"]
        sys.modules.update(saved)


class TestReportFactory(trc.TestResultCollector):
    """
    Tests that report selection degrades to HTML rather than failing when the
    optional PDF stack is absent.
    """

    def setUp(self):
        ReportFactory.WARNING_PRINTED = False

    def tearDown(self):
        ReportFactory.WARNING_PRINTED = False
        patch.stopall()

    def test_model_analyzer_imports_without_pdfkit(self):
        """model_analyzer must stay importable without the optional extra.

        This is the regression that matters: a module-scope ``import pdfkit``
        anywhere in the import graph runs before the PATH probe, so it would
        break every Model Analyzer command and not just reporting.

        Run in a subprocess rather than by reloading in place -- a reload
        rebinds the module's own ``ReportFactory`` while this module keeps the
        original class, and the two then disagree about ``WARNING_PRINTED``.
        """
        script = (
            "import sys\n"
            "sys.modules['pdfkit'] = None\n"
            "import model_analyzer.entrypoint\n"
            "import model_analyzer.reports.report_factory\n"
            "import model_analyzer.reports.report_manager\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_html_is_used_when_the_renderer_is_absent(self):
        with patch("shutil.which", return_value=None):
            self.assertIsInstance(ReportFactory.create_report(), HTMLReport)

    def test_html_is_used_when_the_renderer_is_present_but_pdfkit_is_not(self):
        with patch("shutil.which", return_value="/usr/bin/wkhtmltopdf"):
            with pdfkit_unavailable():
                report = ReportFactory.create_report()
        self.assertIsInstance(report, HTMLReport)
        self.assertNotEqual(type(report).__name__, "PDFReport")

    def test_pdf_is_used_when_both_are_present(self):
        if importlib.util.find_spec("pdfkit") is None:
            self.skipTest("requires the optional pdfkit dependency")
        with patch("shutil.which", return_value="/usr/bin/wkhtmltopdf"):
            report = ReportFactory.create_report()
        self.assertEqual(type(report).__name__, "PDFReport")

    def test_the_fallback_warning_is_printed_once(self):
        with patch("shutil.which", return_value=None):
            with self.assertLogs(level="WARNING") as captured:
                ReportFactory.create_report()
                ReportFactory.create_report()
        self.assertEqual(len(captured.records), 1)
        self.assertIn("wkhtmltopdf is not installed", captured.records[0].getMessage())

    def test_the_missing_pdfkit_warning_names_the_extra(self):
        with patch("shutil.which", return_value="/usr/bin/wkhtmltopdf"):
            with pdfkit_unavailable():
                with self.assertLogs(level="WARNING") as captured:
                    ReportFactory.create_report()
        self.assertIn("triton-model-analyzer[pdf]", captured.records[0].getMessage())


if __name__ == "__main__":
    unittest.main()
