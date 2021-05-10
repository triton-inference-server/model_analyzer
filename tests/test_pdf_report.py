# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import MagicMock, patch, mock_open
import base64

from matplotlib.pyplot import table

from .common import test_result_collector as trc
from model_analyzer.result.result_table import ResultTable
from model_analyzer.reports.pdf_report import PDFReport


class TestPDFReportMethods(trc.TestResultCollector):
    """
    Tests the methods of the PDFReport class
    """
    def setUp(self):
        self.maxDiff = None
        self.report = PDFReport()

    def test_add_title(self):
        self.report.add_title('Test PDF Report')
        expected_report_body = '<html><head><style></style></head><body><center><h1>Test PDF Report</h1></center></body></html>'
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_subheading(self):
        # Add one subheading
        self.report.add_subheading('Throughput vs. Latency')
        expected_report_body = '<html><head><style></style></head><body><h3>Throughput vs. Latency</h3></body></html>'
        self.assertEqual(self.report.document(), expected_report_body)

        # Add another subheading
        self.report.add_subheading('GPU Memory vs. Latency')
        expected_report_body = (
            "<html><head><style></style></head><body><h3>Throughput vs. Latency</h3>"
            "<h3>GPU Memory vs. Latency</h3></body></html>")
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_image(self):
        with patch('model_analyzer.reports.pdf_report.open',
                   mock_open(read_data=bytes('>:('.encode('ascii')))):
            self.report.add_images(['test_image_file'], ['test_caption'])
        img_content = base64.b64encode(bytes(
            '>:('.encode('ascii'))).decode('ascii')
        expected_report_body = (
            f'<html><head><style></style></head><body><center><div><div class="image" style="float:center;width:100%"><img src="data:image/png;base64,{img_content}"'
            ' style="width:100%"><center><div style="font-weight:bold;font-size:12;padding-bottom:20px">'
            'test_caption</div></center></div></div></center></body></html>')
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_paragraph(self):
        test_paragraph = ("This is a test paragraph with a lot to say."
                          " There is more than one line in this paragraph.")

        # Default font size
        self.report.add_paragraph(test_paragraph)
        expected_report_body = (
            "<html><head><style></style></head><body><div style=\"font-size:14\">"
            f"<p>{test_paragraph}</p></div></body></html>")
        self.assertEqual(self.report.document(), expected_report_body)

        # custom font size
        self.report.add_paragraph(test_paragraph, font_size=20)
        expected_report_body = (
            "<html><head><style></style></head><body><div style=\"font-size:14\">"
            f"<p>{test_paragraph}</p></div><div style=\"font-size:20\">"
            f"<p>{test_paragraph}</p></div></body></html>")
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_table(self):
        result_table = ResultTable(['header1', 'header2'])
        # Try empty table
        self.report.add_table(table=result_table)
        table_style = "border: 1px solid black;border-collapse: collapse;text-align: center;width: 80%;padding: 5px 10px;font-size: 11pt"
        expected_report_body = ("<html><head><style></style></head><body>"
                                f"<center><table style=\"{table_style}\">"
                                "<tr>"
                                f"<th style=\"{table_style}\">header1</th>"
                                f"<th style=\"{table_style}\">header2</th>"
                                "</tr>"
                                "</table></center>"
                                "</body></html>")
        self.assertEqual(self.report.document(), expected_report_body)

        # Fill table
        for i in range(2):
            result_table.insert_row_by_index([f'value{i}1', f'value{i}2'])

        # Table has 5 rows
        self.report.add_table(table=result_table)
        expected_report_body = ("<html><head><style></style></head><body>"
                                f"<center><table style=\"{table_style}\">"
                                "<tr>"
                                f"<th style=\"{table_style}\">header1</th>"
                                f"<th style=\"{table_style}\">header2</th>"
                                "</tr>"
                                "</table></center>"
                                f"<center><table style=\"{table_style}\">"
                                "<tr>"
                                f"<th style=\"{table_style}\">header1</th>"
                                f"<th style=\"{table_style}\">header2</th>"
                                "</tr>"
                                "<tr>"
                                f"<td style=\"{table_style}\">value01</td>"
                                f"<td style=\"{table_style}\">value02</td>"
                                "</tr>"
                                "<tr>"
                                f"<td style=\"{table_style}\">value11</td>"
                                f"<td style=\"{table_style}\">value12</td>"
                                "</tr>"
                                "</table></center>"
                                "</body></html>")

        self.assertEqual(self.report.document(), expected_report_body)

    def test_write_report(self):
        with patch('model_analyzer.reports.pdf_report.pdfkit',
                   MagicMock()) as pdfkit_mock:
            self.report.add_title('Test PDF Report')
            self.report.add_subheading('Throughput vs. Latency')
            test_paragraph = (
                "This is a test paragraph with a lot to say."
                " There is more than one line in this paragraph.")
            self.report.add_paragraph(test_paragraph, font_size=14)
            self.report.write_report('test_report_filename')

            expected_report_body = (
                "<html><head><style></style></head><body><center><h1>Test PDF Report</h1></center><h3>Throughput vs. Latency</h3>"
                f"<div style=\"font-size:14\"><p>{test_paragraph}</p></div>"
                "</body></html>")
            pdfkit_mock.from_string.assert_called_with(expected_report_body,
                                                       'test_report_filename')
