#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import base64
import unittest
from unittest.mock import mock_open, patch

from model_analyzer.reports.html_report import HTMLReport, Markup
from model_analyzer.result.result_table import ResultTable

from .common import test_result_collector as trc


class TestHTMLReportMethods(trc.TestResultCollector):
    """
    Tests the methods of the HTMLReport class
    """

    def setUp(self):
        self.maxDiff = None
        self.report = HTMLReport()

    def tearDown(self):
        patch.stopall()

    def test_add_title(self):
        self.report.add_title("Test HTML Report")
        expected_report_body = "<html><head><style></style></head><body><center><h1>Test HTML Report</h1></center></body></html>"
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_subheading(self):
        # Add one subheading
        self.report.add_subheading("Throughput vs. Latency")
        expected_report_body = "<html><head><style></style></head><body><h3>Throughput vs. Latency</h3></body></html>"
        self.assertEqual(self.report.document(), expected_report_body)

        # Add another subheading
        self.report.add_subheading("GPU Memory vs. Latency")
        expected_report_body = (
            "<html><head><style></style></head><body><h3>Throughput vs. Latency</h3>"
            "<h3>GPU Memory vs. Latency</h3></body></html>"
        )
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_image(self):
        with patch(
            "model_analyzer.reports.html_report.open",
            mock_open(read_data=bytes(">:(".encode("ascii"))),
        ):
            self.report.add_images(["test_image_file"], ["test_caption"])
        img_content = base64.b64encode(bytes(">:(".encode("ascii"))).decode("ascii")
        expected_report_body = (
            f'<html><head><style></style></head><body><center><div><div class="image" style="float:center;width:100%"><img src="data:image/png;base64,{img_content}"'
            ' style="width:100%"><center><div style="font-weight:bold;font-size:12;padding-bottom:20px">'
            "test_caption</div></center></div></div></center></body></html>"
        )
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_paragraph(self):
        test_paragraph = (
            "This is a test paragraph with a lot to say."
            " There is more than one line in this paragraph."
        )

        # Default font size
        self.report.add_paragraph(test_paragraph)
        expected_report_body = (
            '<html><head><style></style></head><body><div style="font-size:14">'
            f"<p>{test_paragraph}</p></div></body></html>"
        )
        self.assertEqual(self.report.document(), expected_report_body)

        # custom font size
        self.report.add_paragraph(test_paragraph, font_size=20)
        expected_report_body = (
            '<html><head><style></style></head><body><div style="font-size:14">'
            f'<p>{test_paragraph}</p></div><div style="font-size:20">'
            f"<p>{test_paragraph}</p></div></body></html>"
        )
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_table(self):
        result_table = ResultTable(["header1", "header2"])
        # Try empty table
        self.report.add_table(table=result_table)
        table_style = "border: 1px solid black;border-collapse: collapse;text-align: center;width: 80%;padding: 5px 10px;font-size: 11pt"
        expected_report_body = (
            "<html><head><style></style></head><body>"
            f'<center><table style="{table_style}">'
            "<tr>"
            f'<th style="{table_style}">header1</th>'
            f'<th style="{table_style}">header2</th>'
            "</tr>"
            "</table></center>"
            "</body></html>"
        )
        self.assertEqual(self.report.document(), expected_report_body)

        # Fill table
        for i in range(2):
            result_table.insert_row_by_index([f"value{i}1", f"value{i}2"])

        # Table has 5 rows
        self.report.add_table(table=result_table)
        expected_report_body = (
            "<html><head><style></style></head><body>"
            f'<center><table style="{table_style}">'
            "<tr>"
            f'<th style="{table_style}">header1</th>'
            f'<th style="{table_style}">header2</th>'
            "</tr>"
            "</table></center>"
            f'<center><table style="{table_style}">'
            "<tr>"
            f'<th style="{table_style}">header1</th>'
            f'<th style="{table_style}">header2</th>'
            "</tr>"
            "<tr>"
            f'<td style="{table_style}">value01</td>'
            f'<td style="{table_style}">value02</td>'
            "</tr>"
            "<tr>"
            f'<td style="{table_style}">value11</td>'
            f'<td style="{table_style}">value12</td>'
            "</tr>"
            "</table></center>"
            "</body></html>"
        )

        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_title_escapes_markup(self):
        """
        A model name reaching the report must not be able to close the tag
        it sits in. NVBug 6833957.
        """

        self.report.add_title("<script>alert(1)</script>")
        document = self.report.document()
        self.assertNotIn("<script>", document)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", document)

    def test_add_subheading_escapes_markup(self):
        self.report.add_subheading("Model Config: <script>alert(1)</script>")
        document = self.report.document()
        self.assertNotIn("<script>", document)
        self.assertIn("&lt;script&gt;", document)

    def test_add_table_escapes_headers_and_cells(self):
        """
        Every value in a table is data, never markup, so both headers and
        cells are escaped.
        """

        table = ResultTable(headers=["Model", "<script>alert(1)</script>"], title="t")
        table.insert_row_by_index(["<img src=x onerror=alert(1)>", "7"])
        self.report.add_table(table=table)

        document = self.report.document()
        self.assertNotIn("<script>", document)
        self.assertNotIn("<img", document)
        self.assertIn("&lt;script&gt;", document)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", document)

    def test_add_images_escapes_caption(self):
        with patch(
            "model_analyzer.reports.html_report.open",
            mock_open(read_data=bytes(">:(".encode("ascii"))),
        ):
            self.report.add_images(["f"], ["<script>alert(1)</script>"])

        document = self.report.document()
        self.assertNotIn("<script>", document)
        self.assertIn("&lt;script&gt;", document)

    def test_add_table_preserves_marked_markup_cells(self):
        """
        A few cells are markup the report composed itself -- config names
        joined with <br>, a metric wrapped in <strong>. Those arrive as
        Markup and must pass through add_table untouched, or the separators
        and emphasis render as visible tag text.
        """

        table = ResultTable(headers=["Models", "Throughput"], title="t")
        table.insert_row_by_index(
            [Markup("alpha<br>beta"), Markup("<strong>7</strong> (7, 8)")]
        )
        self.report.add_table(table=table)

        document = self.report.document()
        self.assertIn("alpha<br>beta", document)
        self.assertIn("<strong>7</strong>", document)
        self.assertNotIn("&lt;br&gt;", document)

    def test_add_table_escapes_marked_cell_contents_exactly_once(self):
        """
        Regression: a Markup cell has already escaped its own substitutions,
        so add_table must not escape it again. Double escaping turned a model
        named a<b>c into the visible text a&lt;b&gt;c -- the same class of
        defect as the unescaped name it replaced, since the reader is shown
        something that is not the model's name.
        """

        table = ResultTable(headers=["Models"], title="t")
        table.insert_row_by_index([Markup("a&lt;b&gt;c<br>d&lt;e&gt;f")])
        self.report.add_table(table=table)

        document = self.report.document()
        self.assertIn("a&lt;b&gt;c", document)
        self.assertNotIn("&amp;lt;", document)
        self.assertNotIn("<b>", document)

    def test_escape_is_idempotent_only_through_markup(self):
        """
        Plain text is always escaped; only an explicit Markup marker opts out.
        A fragment someone forgets to mark shows its tags as text, which is
        visible in review, rather than running them.
        """

        from model_analyzer.reports.html_report import escape

        self.assertEqual(escape("a<b>c"), "a&lt;b&gt;c")
        self.assertEqual(escape(Markup("a<br>b")), "a<br>b")
        self.assertEqual(escape(escape("a<b>c")), "a&amp;lt;b&amp;gt;c")

    def test_add_paragraph_preserves_intentional_markup(self):
        """
        report_manager.py builds its sentences with <strong> and <br> on
        purpose, so paragraphs must NOT be escaped at the sink. Escaping
        here would render every summary report's own markup as literal
        text; those sentences escape their substitutions individually.
        """

        sentence = (
            "is <strong>23%</strong> better than the default configuration"
            " at meeting the objectives"
        )
        self.report.add_paragraph(sentence)
        self.assertIn("<strong>23%</strong>", self.report.document())

    def test_write_report(self):
        self.report.add_title("Test HTML Report")
        self.report.add_subheading("Throughput vs. Latency")
        test_paragraph = (
            "This is a test paragraph with a lot to say."
            " There is more than one line in this paragraph."
        )
        self.report.add_paragraph(test_paragraph, font_size=14)

        expected_report_body = (
            "<html><head><style></style></head><body><center><h1>Test HTML Report</h1></center><h3>Throughput vs. Latency</h3>"
            f'<div style="font-size:14"><p>{test_paragraph}</p></div>'
            "</body></html>"
        )

        with patch("builtins.open", mock_open()) as mocked_file:
            self.report.write_report("test_report_filename")
        mocked_file.assert_called_once_with(f"test_report_filename", "w")
        mocked_file().write.assert_called_once_with(f"{expected_report_body}")


if __name__ == "__main__":
    unittest.main()
