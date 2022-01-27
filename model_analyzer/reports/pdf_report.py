# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .html_report import HTMLReport
import pdfkit


class PDFReport(HTMLReport):
    """
    A report that takes
    an html report and converts
    it to PDF
    """

    def __init__(self, html_report=None):
        super().__init__()
        if html_report is not None:
            self._head = html_report._head
            self._body = html_report._body

    def write_report(self, filename):
        """
        Write the report to disk with
        filename

        Parameters
        ----------
        filename : str
            The name of the report
        """

        pdfkit.from_string(self.document(),
                           f'{filename}',
                           options={'quiet': ''})

    def get_file_extension(self):
        """
        Return the file extension for 
        the type of report
        """
        return "pdf"
