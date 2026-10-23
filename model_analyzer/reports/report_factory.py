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

import logging

from model_analyzer.constants import LOGGER_NAME

from .html_report import HTMLReport

logger = logging.getLogger(LOGGER_NAME)


class ReportFactory:
    """
    Factory that returns the correct report object
    """

    PDF_PACKAGE = "weasyprint"
    WARNING_PRINTED = False

    @staticmethod
    def create_report():
        try:
            return ReportFactory._create_pdf_report()
        except (ImportError, OSError) as error:
            # ImportError: the "pdf" extra is not installed, or an older
            # WeasyPrint without the URLFetcher this module needs.
            # OSError: WeasyPrint is installed but its native libraries
            # (Pango, HarfBuzz, fontconfig) are not -- importing it then
            # fails with "cannot load library", not ImportError. Both mean
            # the same thing to a user: no PDF renderer, so emit HTML.
            ReportFactory._warn_once(
                f"Warning: html reports are being generated instead of pdf "
                f"because {ReportFactory.PDF_PACKAGE} is not usable ({error}). "
                f"If you want pdf reports, install Model Analyzer with its pdf "
                f'extra: "pip install triton-model-analyzer[pdf]"'
            )
            return ReportFactory._create_html_report()

    @staticmethod
    def _warn_once(message):
        if not ReportFactory.WARNING_PRINTED:
            ReportFactory.WARNING_PRINTED = True
            logging.warning(message)

    @staticmethod
    def _create_pdf_report():
        # Imported here, not at module scope: weasyprint is an optional
        # dependency and model_analyzer must stay importable without it. An
        # import at the top of this module would run before the guard in
        # create_report, which would make the HTML fallback unreachable and
        # break every command, not just report generation.
        from .pdf_report import PDFReport

        return PDFReport()

    @staticmethod
    def _create_html_report():
        return HTMLReport()
