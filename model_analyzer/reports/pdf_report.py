#!/usr/bin/env python3

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

import pdfkit

from .html_report import HTMLReport


class PDFReport(HTMLReport):
    """
    A report that takes
    an html report and converts
    it to PDF
    """

    def __init__(self, html_report=None):
        super().__init__(html_report)

    def write_report(self, filename):
        """
        Write the report to disk with
        filename

        Parameters
        ----------
        filename : str
            The name of the report
        """

        pdfkit.from_string(self.document(), f"{filename}", options={"quiet": ""})

    def get_file_extension(self):
        """
        Return the file extension for
        the type of report
        """
        return "pdf"
