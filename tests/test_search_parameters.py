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

import unittest
from unittest.mock import MagicMock, patch

from model_analyzer.config.generate.search_parameters import (
    ParameterCategory,
    ParameterUsage,
    SearchParameters,
)
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from tests.test_config import TestConfig

from .common import test_result_collector as trc
from .mocks.mock_os import MockOSMethods


class TestSearchParameters(trc.TestResultCollector):
    def setUp(self):
        # Mock path validation
        self.mock_os = MockOSMethods(
            mock_paths=["model_analyzer.config.input.config_utils"]
        )
        self.mock_os.start()

        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "-f",
            "path-to-config-file",
            "--run-config-search-mode",
            "optuna",
        ]

        yaml_content = """
        profile_models: add_sub
        """

        config = TestConfig()._evaluate_config(args=args, yaml_content=yaml_content)

        self.search_parameters = SearchParameters(config)

        self.search_parameters._add_search_parameter(
            name="concurrency",
            usage=ParameterUsage.RUNTIME,
            category=ParameterCategory.EXPONENTIAL,
            min_range=0,
            max_range=10,
        )

        self.search_parameters._add_search_parameter(
            name="instance_count",
            usage=ParameterUsage.MODEL,
            category=ParameterCategory.INTEGER,
            min_range=1,
            max_range=8,
        )

        self.search_parameters._add_search_parameter(
            name="size",
            usage=ParameterUsage.BUILD,
            category=ParameterCategory.LIST,
            enumerated_list=["FP8", "FP16", "FP32"],
        )

    def tearDown(self):
        self.mock_os.stop()
        patch.stopall()

    def test_exponential_parameter(self):
        """
        Test exponential parameter, accessing dataclass directly
        """

        parameter = self.search_parameters.get_parameter("concurrency")

        self.assertEqual(ParameterUsage.RUNTIME, parameter.usage)
        self.assertEqual(ParameterCategory.EXPONENTIAL, parameter.category)
        self.assertEqual(0, parameter.min_range)
        self.assertEqual(10, parameter.max_range)

    def test_integer_parameter(self):
        """
        Test integer parameter, using accessor methods
        """

        self.assertEqual(
            ParameterUsage.MODEL,
            self.search_parameters.get_type("instance_count"),
        )
        self.assertEqual(
            ParameterCategory.INTEGER,
            self.search_parameters.get_category("instance_count"),
        )
        self.assertEqual((1, 8), self.search_parameters.get_range("instance_count"))

    def test_list_parameter(self):
        """
        Test list parameter, using accessor methods
        """

        self.assertEqual(
            ParameterUsage.BUILD,
            self.search_parameters.get_type("size"),
        )
        self.assertEqual(
            ParameterCategory.LIST,
            self.search_parameters.get_category("size"),
        )
        self.assertEqual(
            ["FP8", "FP16", "FP32"], self.search_parameters.get_list("size")
        )

    def test_illegal_inputs(self):
        """
        Check that an exception is raised for illegal input combos
        """
        with self.assertRaises(TritonModelAnalyzerException):
            self.search_parameters._add_search_parameter(
                name="concurrency",
                usage=ParameterUsage.RUNTIME,
                category=ParameterCategory.EXPONENTIAL,
                max_range=10,
            )

        with self.assertRaises(TritonModelAnalyzerException):
            self.search_parameters._add_search_parameter(
                name="concurrency",
                usage=ParameterUsage.RUNTIME,
                category=ParameterCategory.EXPONENTIAL,
                min_range=0,
            )

        with self.assertRaises(TritonModelAnalyzerException):
            self.search_parameters._add_search_parameter(
                name="concurrency",
                usage=ParameterUsage.RUNTIME,
                category=ParameterCategory.EXPONENTIAL,
                min_range=10,
                max_range=9,
            )

        with self.assertRaises(TritonModelAnalyzerException):
            self.search_parameters._add_search_parameter(
                name="size",
                usage=ParameterUsage.BUILD,
                category=ParameterCategory.LIST,
            )

        with self.assertRaises(TritonModelAnalyzerException):
            self.search_parameters._add_search_parameter(
                name="size",
                usage=ParameterUsage.BUILD,
                category=ParameterCategory.LIST,
                enumerated_list=["FP8", "FP16", "FP32"],
                min_range=0,
            )

        with self.assertRaises(TritonModelAnalyzerException):
            self.search_parameters._add_search_parameter(
                name="size",
                usage=ParameterUsage.BUILD,
                category=ParameterCategory.LIST,
                enumerated_list=["FP8", "FP16", "FP32"],
                max_range=10,
            )


if __name__ == "__main__":
    unittest.main()
