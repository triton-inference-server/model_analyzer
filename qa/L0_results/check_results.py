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

import argparse
import os
import sys

import yaml


def report_extension():
    """
    Returns the file extension Model Analyzer uses for reports.

    ReportFactory emits PDF reports only when WeasyPrint is usable, so the
    expected artifact depends on the image the tests run in.

    The exact names pdf_report imports are imported here, not just the
    package: a WeasyPrint older than 70 has no URLFetcher, so PDFReport
    fails to import and HTML is produced, and a bare ``import weasyprint``
    would still say "pdf" and reject that correct output. The import is
    attempted rather than looked up with find_spec for the same reason --
    WeasyPrint installed without its native libraries imports with OSError,
    which ReportFactory also treats as "no renderer". This check and that
    one must agree.
    """

    try:
        from weasyprint import HTML  # noqa: F401
        from weasyprint.urls import URLFetcher  # noqa: F401
    except (ImportError, OSError):
        return "html"
    return "pdf"


class TestOutputValidator:
    """
    Functions that validate the output of
    the test
    """

    def __init__(self, config, test_name, export_path):
        self._config = config
        self._export_path = export_path
        check_function = self.__getattribute__(f"check_{test_name}")

        if check_function():
            sys.exit(0)
        else:
            sys.exit(1)

    def check_summaries(self):
        """
        Checks if the result summary reports exist
        in the required location

        Returns
        -------
        True if all exist, False otherwise
        """

        for model in self._config["profile_models"]:
            if not os.path.exists(
                os.path.join(
                    self._export_path,
                    "reports",
                    "summaries",
                    model,
                    f"result_summary.{report_extension()}",
                )
            ):
                print(f"\n***\n*** Summary not found for {model}.\n***")
                return False

        # First check for the best models report
        profile_models = set(self._config["profile_models"])
        report_dirs = set(
            os.listdir(os.path.join(self._export_path, "reports", "summaries"))
        )
        if len(report_dirs - profile_models) != 1:
            print("\n***\n*** Top models summary not found.\n***")
            return False

        # Should be only 1 element in set difference
        for dir in report_dirs - profile_models:
            if not os.path.exists(
                os.path.join(
                    self._export_path,
                    "reports",
                    "summaries",
                    dir,
                    f"result_summary.{report_extension()}",
                )
            ):
                print("\n***\n*** Top models summary not found.\n***")
                return False

        return True

    def check_detailed_reports(self):
        """
        Checks if the detailed reports exist
        in the required location
        """

        for model_config in self._config["report_model_configs"]:
            if not os.path.exists(
                os.path.join(
                    self._export_path,
                    "reports",
                    "detailed",
                    model_config,
                    f"detailed_report.{report_extension()}",
                )
            ):
                print(f"\n***\n*** Detailed report not found for {model_config}.\n***")
                return False
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--config-file",
        type=str,
        required=True,
        help="The config file for this test",
    )
    parser.add_argument(
        "-d",
        "--export-path",
        type=str,
        required=True,
        help="The full path to the export directory",
    )
    parser.add_argument(
        "-t",
        "--test-name",
        type=str,
        required=True,
        help="The name of the test to be run.",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    TestOutputValidator(config, args.test_name, args.export_path)
