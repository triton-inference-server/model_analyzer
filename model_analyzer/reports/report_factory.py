# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .pdf_report import PDFReport
from .html_report import HTMLReport
import apt


class ReportFactory:
    """
    Factory that returns the correct report object
    """

    PDF_PACKAGE = "wkhtmltopdf"

    @staticmethod
    def create_report():
        if ReportFactory._is_apt_package_installed(
                f"{ReportFactory.PDF_PACKAGE}"):
            return ReportFactory.create_pdf_report()
        else:
            print(f"Warning: {ReportFactory.PDF_PACKAGE} is not installed. Pdf reports cannot be generated. "\
                f"Html reports will be generated instead."
                )
            return ReportFactory.create_html_report()

    @staticmethod
    def create_pdf_report():
        return PDFReport()

    @staticmethod
    def create_html_report():
        return HTMLReport()

    @staticmethod
    def _is_apt_package_installed(package_name):
        cache = apt.Cache()
        return cache[f"{package_name}"].is_installed
