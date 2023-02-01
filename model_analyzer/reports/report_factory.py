# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.constants import LOGGER_NAME
from .pdf_report import PDFReport
from .html_report import HTMLReport
import logging
import shutil

logger = logging.getLogger(LOGGER_NAME)


class ReportFactory:
    """
    Factory that returns the correct report object
    """

    PDF_PACKAGE = "wkhtmltopdf"
    WARNING_PRINTED = False

    @staticmethod
    def create_report():
        if ReportFactory._is_package_installed(f"{ReportFactory.PDF_PACKAGE}"):
            return ReportFactory._create_pdf_report()
        else:
            if not ReportFactory.WARNING_PRINTED:
                ReportFactory.WARNING_PRINTED = True
                logging.warning(
                    f'Warning: html reports are being generated instead of pdf because '
                    f'{ReportFactory.PDF_PACKAGE} is not installed. If you want pdf '
                    f'reports, run the following command and then rerun Model Analyzer: '
                    f'"sudo apt-get update && sudo apt-get install wkhtmltopdf"'
                )
            return ReportFactory._create_html_report()

    @staticmethod
    def _create_pdf_report():
        return PDFReport()

    @staticmethod
    def _create_html_report():
        return HTMLReport()

    @staticmethod
    def _is_package_installed(package_name):
        package_found = shutil.which(f"{package_name}")
        return package_found is not None
