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
from math import log2
from unittest.mock import MagicMock, patch

import model_analyzer.config.input.config_defaults as default
from model_analyzer.analyzer import Analyzer
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

    def test_search_parameter_creation_default(self):
        """
        Test that search parameters are correctly created in default optuna case
        """

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

        analyzer = Analyzer(config, MagicMock(), MagicMock(), MagicMock())
        analyzer._populate_search_parameters()

        # batch_sizes
        batch_sizes = analyzer._search_parameters["add_sub"].get_parameter(
            "batch_sizes"
        )
        self.assertEqual(ParameterUsage.MODEL, batch_sizes.usage)
        self.assertEqual(ParameterCategory.EXPONENTIAL, batch_sizes.category)
        self.assertEqual(
            log2(default.DEFAULT_RUN_CONFIG_MIN_MODEL_BATCH_SIZE), batch_sizes.min_range
        )
        self.assertEqual(
            log2(default.DEFAULT_RUN_CONFIG_MAX_MODEL_BATCH_SIZE), batch_sizes.max_range
        )

        # concurrency
        concurrency = analyzer._search_parameters["add_sub"].get_parameter(
            "concurrency"
        )
        self.assertEqual(ParameterUsage.RUNTIME, concurrency.usage)
        self.assertEqual(ParameterCategory.EXPONENTIAL, concurrency.category)
        self.assertEqual(
            log2(default.DEFAULT_RUN_CONFIG_MIN_CONCURRENCY), concurrency.min_range
        )
        self.assertEqual(
            log2(default.DEFAULT_RUN_CONFIG_MAX_CONCURRENCY), concurrency.max_range
        )

        # instance_group
        instance_group = analyzer._search_parameters["add_sub"].get_parameter(
            "instance_group"
        )
        self.assertEqual(ParameterUsage.MODEL, instance_group.usage)
        self.assertEqual(ParameterCategory.INTEGER, instance_group.category)
        self.assertEqual(
            default.DEFAULT_RUN_CONFIG_MIN_INSTANCE_COUNT, instance_group.min_range
        )
        self.assertEqual(
            default.DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT, instance_group.max_range
        )

    def test_search_parameter_creation_multi_model_non_default(self):
        """
        Test that search parameters are correctly created in
        a multi-model non-default optuna case
        """

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
        run_config_search_mode: optuna
        profile_models:
            add_sub:
                parameters:
                    batch_sizes: [16, 32, 64]
                model_config_parameters:
                    dynamic_batching:
                        max_queue_delay_microseconds: [100, 200, 300]
                    instance_group:
                        - kind: KIND_GPU
                          count: [1, 2, 3, 4]
            mult_div:
                parameters:
                    concurrency: [1, 8, 64, 256]
        """

        config = TestConfig()._evaluate_config(args, yaml_content)

        analyzer = Analyzer(config, MagicMock(), MagicMock(), MagicMock())
        analyzer._populate_search_parameters()

        # ===================================================================
        # ADD_SUB
        # ===================================================================

        # batch_sizes
        # ===================================================================
        batch_sizes = analyzer._search_parameters["add_sub"].get_parameter(
            "batch_sizes"
        )
        self.assertEqual(ParameterUsage.MODEL, batch_sizes.usage)
        self.assertEqual(ParameterCategory.LIST, batch_sizes.category)
        self.assertEqual([16, 32, 64], batch_sizes.enumerated_list)

        # concurrency
        # ===================================================================
        concurrency = analyzer._search_parameters["add_sub"].get_parameter(
            "concurrency"
        )
        self.assertEqual(ParameterUsage.RUNTIME, concurrency.usage)
        self.assertEqual(ParameterCategory.EXPONENTIAL, concurrency.category)
        self.assertEqual(
            log2(default.DEFAULT_RUN_CONFIG_MIN_CONCURRENCY), concurrency.min_range
        )
        self.assertEqual(
            log2(default.DEFAULT_RUN_CONFIG_MAX_CONCURRENCY), concurrency.max_range
        )

        # instance_group
        # ===================================================================
        instance_group = analyzer._search_parameters["add_sub"].get_parameter(
            "instance_group"
        )
        self.assertEqual(ParameterUsage.MODEL, instance_group.usage)
        self.assertEqual(ParameterCategory.LIST, instance_group.category)
        self.assertEqual([1, 2, 3, 4], instance_group.enumerated_list)

        instance_group = analyzer._search_parameters["add_sub"].get_parameter(
            "max_queue_delay_microseconds"
        )
        self.assertEqual(ParameterUsage.MODEL, instance_group.usage)
        self.assertEqual(ParameterCategory.LIST, instance_group.category)
        self.assertEqual([100, 200, 300], instance_group.enumerated_list)

        # ===================================================================
        # MULT_DIV
        # ===================================================================

        # batch_sizes
        # ===================================================================
        batch_sizes = analyzer._search_parameters["mult_div"].get_parameter(
            "batch_sizes"
        )
        self.assertEqual(ParameterUsage.MODEL, batch_sizes.usage)
        self.assertEqual(ParameterCategory.EXPONENTIAL, batch_sizes.category)
        self.assertEqual(
            log2(default.DEFAULT_RUN_CONFIG_MIN_MODEL_BATCH_SIZE), batch_sizes.min_range
        )
        self.assertEqual(
            log2(default.DEFAULT_RUN_CONFIG_MAX_MODEL_BATCH_SIZE), batch_sizes.max_range
        )

        # concurrency
        # ===================================================================
        concurrency = analyzer._search_parameters["mult_div"].get_parameter(
            "concurrency"
        )
        self.assertEqual(ParameterUsage.RUNTIME, concurrency.usage)
        self.assertEqual(ParameterCategory.LIST, concurrency.category)
        self.assertEqual([1, 8, 64, 256], concurrency.enumerated_list)

        # instance_group
        # ===================================================================
        instance_group = analyzer._search_parameters["mult_div"].get_parameter(
            "instance_group"
        )
        self.assertEqual(ParameterUsage.MODEL, instance_group.usage)
        self.assertEqual(ParameterCategory.INTEGER, instance_group.category)
        self.assertEqual(
            default.DEFAULT_RUN_CONFIG_MIN_INSTANCE_COUNT, instance_group.min_range
        )
        self.assertEqual(
            default.DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT, instance_group.max_range
        )


if __name__ == "__main__":
    unittest.main()
