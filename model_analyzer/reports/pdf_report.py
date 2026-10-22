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

from weasyprint import HTML
from weasyprint.urls import URLFetcher

from .html_report import HTMLReport


class PDFReport(HTMLReport):
    """
    A report that takes
    an html report and converts
    it to PDF
    """

    # WeasyPrint is a layout engine, not a browser: it has no JavaScript
    # interpreter at all, so the scripting exposure that wkhtmltopdf carried
    # through QtWebKit is gone by construction rather than by an opt-out flag.
    #
    # Resource loading is not gone by construction. WeasyPrint's default
    # fetcher reads file:// URLs and issues real HTTP requests for anything
    # the document references. Report images are inlined as data: URIs by
    # html_report, so nothing legitimate needs the filesystem or the network,
    # and a subresource that asks for either got into the markup by accident
    # or by injection. Restricting the fetcher to data: refuses both before
    # the read or the request happens -- a control wkhtmltopdf never offered.
    URL_FETCHER = URLFetcher(allowed_protocols=["data"])

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

        # No base_url: a relative URL then has nothing to resolve against, so
        # the document cannot reach the directory it is rendered from either.
        HTML(string=self.document(), url_fetcher=self.URL_FETCHER).write_pdf(
            f"{filename}"
        )

    def get_file_extension(self):
        """
        Return the file extension for
        the type of report
        """
        return "pdf"
