#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from .common import test_result_collector as trc

# WeasyPrint is the "pdf" extra, and it raises OSError rather than ImportError
# when its native libraries are absent. Skip on either: there is no renderer
# to test in both cases.
try:
    import weasyprint  # noqa: F401

    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    WEASYPRINT_AVAILABLE = False

if WEASYPRINT_AVAILABLE:
    from model_analyzer.reports.pdf_report import PDFReport


@unittest.skipUnless(
    WEASYPRINT_AVAILABLE, "requires the optional weasyprint dependency"
)
class TestPDFReportMethods(trc.TestResultCollector):
    """
    Tests the methods of the PDFReport class
    """

    def setUp(self):
        self.maxDiff = None
        self.report = PDFReport()

    def tearDown(self):
        patch.stopall()

    def test_write_report(self):
        with patch("model_analyzer.reports.pdf_report.HTML", MagicMock()) as html_mock:
            self.report.add_title("Test PDF Report")
            self.report.add_subheading("Throughput vs. Latency")
            test_paragraph = (
                "This is a test paragraph with a lot to say."
                " There is more than one line in this paragraph."
            )
            self.report.add_paragraph(test_paragraph, font_size=14)
            self.report.write_report("test_report_filename")

            expected_report_body = (
                "<html><head><style></style></head><body><center><h1>Test PDF Report</h1></center><h3>Throughput vs. Latency</h3>"
                f'<div style="font-size:14"><p>{test_paragraph}</p></div>'
                "</body></html>"
            )
            self.assertEqual(html_mock.call_args.kwargs["string"], expected_report_body)
            html_mock.return_value.write_pdf.assert_called_with("test_report_filename")

    def test_the_renderer_is_restricted_to_data_uris(self):
        """
        WeasyPrint's default fetcher reads file:// URLs and issues real HTTP
        requests. Report images are inlined as data: URIs, so nothing else
        should ever be fetched.
        """

        with patch("model_analyzer.reports.pdf_report.HTML", MagicMock()) as html_mock:
            self.report.add_title("Test PDF Report")
            self.report.write_report("test_report_filename")

            fetcher = html_mock.call_args.kwargs["url_fetcher"]
            self.assertIs(fetcher, PDFReport.URL_FETCHER)

    def test_no_base_url_is_given_to_the_renderer(self):
        """
        Without a base URL a relative reference has nothing to resolve
        against, so the document cannot reach its own output directory.
        """

        with patch("model_analyzer.reports.pdf_report.HTML", MagicMock()) as html_mock:
            self.report.add_title("Test PDF Report")
            self.report.write_report("test_report_filename")

            self.assertNotIn("base_url", html_mock.call_args.kwargs)

    def test_a_real_render_refuses_file_and_http_subresources(self):
        """
        End-to-end against the real renderer: a document that references a
        local file and a remote image must produce a PDF with neither
        fetched. This is the guarantee wkhtmltopdf could not make -- it had
        no option that blocked remote resource loading.
        """

        with tempfile.TemporaryDirectory() as directory:
            canary_path = os.path.join(directory, "canary.txt")
            with open(canary_path, "w") as canary:
                canary.write("secret-canary-contents")

            report = PDFReport()
            report._body = (
                f'<img src="file://{canary_path}">'
                '<img src="http://localhost:1/remote.png">'
            )
            output = os.path.join(directory, "report.pdf")

            with self.assertLogs("weasyprint", level="WARNING") as captured:
                report.write_report(output)

            self.assertTrue(os.path.exists(output))
            messages = " ".join(record.getMessage() for record in captured.records)
            self.assertIn("disallowed protocol", messages)
            with open(output, "rb") as rendered:
                self.assertNotIn(b"secret-canary-contents", rendered.read())


if __name__ == "__main__":
    unittest.main()
