#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
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

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from .common import test_result_collector as trc

_CHECK_RESULTS_PATH = (
    Path(__file__).resolve().parents[1] / "qa" / "L0_results" / "check_results.py"
)


_ABSENT = object()


@contextmanager
def weasyprint_available(state):
    """Control what ``import weasyprint`` does inside the qa script.

    Three states, because there are three real ones. True: a usable install.
    False: no "pdf" extra. "old": WeasyPrint present but older than the
    URLFetcher pdf_report needs, so PDFReport does not import and HTML is
    produced -- the gate has to say "html" for that case too, or it rejects
    correct output.
    """

    names = ("weasyprint", "weasyprint.urls")
    saved = {name: sys.modules.get(name, _ABSENT) for name in names}

    if state is False:
        sys.modules["weasyprint"] = None
        sys.modules.pop("weasyprint.urls", None)
    else:
        package = types.ModuleType("weasyprint")
        package.HTML = object()
        urls = types.ModuleType("weasyprint.urls")
        if state is True:
            urls.URLFetcher = object()
        package.urls = urls
        sys.modules["weasyprint"] = package
        sys.modules["weasyprint.urls"] = urls
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is _ABSENT:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_check_results():
    spec = importlib.util.spec_from_file_location(
        "qa_check_results", _CHECK_RESULTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestQACheckResults(trc.TestResultCollector):
    """
    The L0_results gate asserts the report artifacts Model Analyzer wrote.
    ReportFactory only emits PDFs when WeasyPrint is usable, and whether it
    is installed depends on the image, so the gate has to follow the same
    condition instead of hardcoding .pdf.
    """

    def setUp(self):
        self.check_results = _load_check_results()
        self.config = {
            "profile_models": ["densenet_onnx"],
            "report_model_configs": ["densenet_onnx_config_0"],
        }

    def tearDown(self):
        patch.stopall()

    def _build_export_path(self, export_path, extension):
        for name in ("densenet_onnx", "top_models"):
            directory = os.path.join(export_path, "reports", "summaries", name)
            os.makedirs(directory)
            Path(directory, f"result_summary.{extension}").touch()

        directory = os.path.join(
            export_path, "reports", "detailed", "densenet_onnx_config_0"
        )
        os.makedirs(directory)
        Path(directory, f"detailed_report.{extension}").touch()

    def _run_check(self, test_name, export_path):
        """
        TestOutputValidator exits the process with the check's result, which
        is the contract test.sh relies on, so assert on the exit code.
        """

        with self.assertRaises(SystemExit) as context:
            self.check_results.TestOutputValidator(self.config, test_name, export_path)
        return context.exception.code

    def test_report_extension_pdf_when_weasyprint_present(self):
        with weasyprint_available(True):
            self.assertEqual(self.check_results.report_extension(), "pdf")

    def test_report_extension_html_when_weasyprint_absent(self):
        with weasyprint_available(False):
            self.assertEqual(self.check_results.report_extension(), "html")

    def test_report_extension_html_when_weasyprint_is_too_old(self):
        """
        A WeasyPrint without URLFetcher cannot render: PDFReport fails to
        import and ReportFactory produces HTML. A bare ``import weasyprint``
        would still report "pdf" here and fail the gate on correct output.
        """

        with weasyprint_available("old"):
            self.assertEqual(self.check_results.report_extension(), "html")

    def test_checks_pass_on_html_reports_without_weasyprint(self):
        """
        An image without the pdf extra, so HTML reports are produced. This
        failed before the fix because the gate looked for .pdf.
        """

        with weasyprint_available(False):
            with tempfile.TemporaryDirectory() as export_path:
                self._build_export_path(export_path, "html")
                self.assertEqual(self._run_check("summaries", export_path), 0)
                self.assertEqual(self._run_check("detailed_reports", export_path), 0)

    def test_checks_pass_on_pdf_reports_with_weasyprint(self):
        """
        The pdf extra installed: PDF reports still gate correctly.
        """

        with weasyprint_available(True):
            with tempfile.TemporaryDirectory() as export_path:
                self._build_export_path(export_path, "pdf")
                self.assertEqual(self._run_check("summaries", export_path), 0)
                self.assertEqual(self._run_check("detailed_reports", export_path), 0)

    def test_checks_fail_when_reports_are_missing(self):
        """
        The gate must still fail when nothing was written, otherwise the
        extension fix would have turned it into a no-op.
        """

        with weasyprint_available(False):
            with tempfile.TemporaryDirectory() as export_path:
                os.makedirs(os.path.join(export_path, "reports", "summaries"))
                os.makedirs(os.path.join(export_path, "reports", "detailed"))
                self.assertEqual(self._run_check("summaries", export_path), 1)
                self.assertEqual(self._run_check("detailed_reports", export_path), 1)


if __name__ == "__main__":
    unittest.main()
