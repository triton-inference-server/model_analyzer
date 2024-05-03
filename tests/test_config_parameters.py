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

from model_analyzer.config.generate.config_parameters import (
    ConfigParameters,
    ParameterCategory,
    ParameterType,
)
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

from .common import test_result_collector as trc


class TestConfigParameters(trc.TestResultCollector):
    def setUp(self):
        self.config_parameters = ConfigParameters()

        self.config_parameters._add_parameter(
            name="concurrency",
            ptype=ParameterType.RUNTIME,
            category=ParameterCategory.EXPONENTIAL,
            min_range=0,
            max_range=10,
        )

        self.config_parameters._add_parameter(
            name="instance_count",
            ptype=ParameterType.MODEL,
            category=ParameterCategory.INTEGER,
            min_range=1,
            max_range=8,
        )

        self.config_parameters._add_parameter(
            name="size",
            ptype=ParameterType.BUILD,
            category=ParameterCategory.LIST,
            enumerated_list=["FP8", "FP16", "FP32"],
        )

    def test_exponential_parameter(self):
        """
        Test exponential parameter, accessing dataclass directly
        """

        parameter = self.config_parameters.get_parameter("concurrency")

        self.assertEqual(ParameterType.RUNTIME, parameter.ptype)
        self.assertEqual(ParameterCategory.EXPONENTIAL, parameter.category)
        self.assertEqual(0, parameter.min_range)
        self.assertEqual(10, parameter.max_range)

    def test_integer_parameter(self):
        """
        Test integer parameter, using accessor methods
        """

        self.assertEqual(
            ParameterType.MODEL,
            self.config_parameters.get_type("instance_count"),
        )
        self.assertEqual(
            ParameterCategory.INTEGER,
            self.config_parameters.get_category("instance_count"),
        )
        self.assertEqual((1, 8), self.config_parameters.get_range("instance_count"))

    def test_list_parameter(self):
        """
        Test list parameter, using accessor methods
        """

        self.assertEqual(
            ParameterType.BUILD,
            self.config_parameters.get_type("size"),
        )
        self.assertEqual(
            ParameterCategory.LIST,
            self.config_parameters.get_category("size"),
        )
        self.assertEqual(
            ["FP8", "FP16", "FP32"], self.config_parameters.get_list("size")
        )

    def test_illegal_inputs(self):
        """
        Check that an exception is raised for illegal input combos
        """
        with self.assertRaises(TritonModelAnalyzerException):
            self.config_parameters._add_parameter(
                name="concurrency",
                ptype=ParameterType.RUNTIME,
                category=ParameterCategory.EXPONENTIAL,
                max_range=10,
            )

        with self.assertRaises(TritonModelAnalyzerException):
            self.config_parameters._add_parameter(
                name="concurrency",
                ptype=ParameterType.RUNTIME,
                category=ParameterCategory.EXPONENTIAL,
                min_range=0,
            )

        with self.assertRaises(TritonModelAnalyzerException):
            self.config_parameters._add_parameter(
                name="concurrency",
                ptype=ParameterType.RUNTIME,
                category=ParameterCategory.EXPONENTIAL,
                min_range=10,
                max_range=9,
            )

        with self.assertRaises(TritonModelAnalyzerException):
            self.config_parameters._add_parameter(
                name="size",
                ptype=ParameterType.BUILD,
                category=ParameterCategory.LIST,
            )

        with self.assertRaises(TritonModelAnalyzerException):
            self.config_parameters._add_parameter(
                name="size",
                ptype=ParameterType.BUILD,
                category=ParameterCategory.LIST,
                enumerated_list=["FP8", "FP16", "FP32"],
                min_range=0,
            )

        with self.assertRaises(TritonModelAnalyzerException):
            self.config_parameters._add_parameter(
                name="size",
                ptype=ParameterType.BUILD,
                category=ParameterCategory.LIST,
                enumerated_list=["FP8", "FP16", "FP32"],
                max_range=10,
            )


if __name__ == "__main__":
    unittest.main()
