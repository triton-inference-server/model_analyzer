#!/usr/bin/env python3

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

import logging
import shutil

from model_analyzer.constants import LOGGER_NAME

from .html_report import HTMLReport

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
            try:
                return ReportFactory._create_pdf_report()
            except ImportError:
                # The renderer is on PATH but the optional Python wrapper is
                # not. HTML is still correct output, so warn and fall through
                # rather than failing the run.
                ReportFactory._warn_once(
                    f"Warning: html reports are being generated instead of pdf "
                    f"because {ReportFactory.PDF_PACKAGE} is installed but the "
                    f"optional pdfkit package is not. If you want pdf reports, "
                    f"reinstall Model Analyzer with its pdf extra: "
                    f'"pip install triton-model-analyzer[pdf]"'
                )
        else:
            ReportFactory._warn_once(
                f"Warning: html reports are being generated instead of pdf because "
                f"{ReportFactory.PDF_PACKAGE} is not installed. If you want pdf "
                f"reports, run the following command and then rerun Model Analyzer: "
                f'"sudo apt-get update && sudo apt-get install wkhtmltopdf"'
            )
        return ReportFactory._create_html_report()

    @staticmethod
    def _warn_once(message):
        if not ReportFactory.WARNING_PRINTED:
            ReportFactory.WARNING_PRINTED = True
            logging.warning(message)

    @staticmethod
    def _create_pdf_report():
        # Imported here, not at module scope: pdfkit is an optional dependency
        # and model_analyzer must stay importable without it. An import at the
        # top of this module runs before the PATH probe above, which would make
        # the HTML fallback unreachable and break every command, not just
        # report generation.
        from .pdf_report import PDFReport

        return PDFReport()

    @staticmethod
    def _create_html_report():
        return HTMLReport()

    @staticmethod
    def _is_package_installed(package_name):
        package_found = shutil.which(f"{package_name}")
        return package_found is not None
