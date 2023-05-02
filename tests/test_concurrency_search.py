# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

from .common.test_utils import construct_run_config_measurement, evaluate_mock_config

from model_analyzer.constants import THROUGHPUT_MINIMUM_CONSECUTIVE_CONCURRENCY_TRIES
from model_analyzer.config.input.config_defaults import DEFAULT_RUN_CONFIG_MAX_CONCURRENCY, DEFAULT_RUN_CONFIG_MIN_CONCURRENCY

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.result.concurrency_search import ConcurrencySearch

from math import log2

import unittest
from unittest.mock import MagicMock, patch
from .common import test_result_collector as trc


class TestConcurrencySearch(trc.TestResultCollector):

    def setUp(self):
        self._min_concurrency_index = int(
            log2(DEFAULT_RUN_CONFIG_MIN_CONCURRENCY))
        self._max_concurrency_index = int(
            log2(DEFAULT_RUN_CONFIG_MAX_CONCURRENCY))

        self._concurrencies = []

    def tearDown(self):
        patch.stopall()

    def test_sweep(self):
        """
        Test sweeping concurrency from min to max, when no constraints are present
        and throughput is linearly increasing
        """
        config = self._create_single_model_no_constraints()
        constraint_manager = ConstraintManager(config)
        concurrency_search = ConcurrencySearch(config)

        for concurrency in concurrency_search.search_concurrencies():
            self._concurrencies.append(concurrency)

            concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=100 * concurrency,
                    latency=10,
                    concurrency=concurrency,
                    constraint_manager=constraint_manager))

        expected_concurrencies = [
            2**c for c in range(self._min_concurrency_index,
                                self._max_concurrency_index + 1)
        ]

        self.assertEqual(self._concurrencies, expected_concurrencies)

    def test_saturating_sweep(self):
        """
        Test sweeping concurrency from min to max, when no constraints are present
        and throughput increases and then saturates
        """
        config = self._create_single_model_no_constraints()
        constraint_manager = ConstraintManager(config)
        concurrency_search = ConcurrencySearch(config)

        # [100, 200, 400, 800, 1000, 1000,...]
        throughputs = [
            100 * 2**c if c < 4 else 1000 for c in range(
                self._min_concurrency_index, self._max_concurrency_index + 1)
        ]

        for i, concurrency in enumerate(
                concurrency_search.search_concurrencies()):
            self._concurrencies.append(concurrency)

            concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=throughputs[i],
                    latency=10,
                    concurrency=concurrency,
                    constraint_manager=constraint_manager))

        expected_concurrencies = [
            2**c
            for c in range(4 + THROUGHPUT_MINIMUM_CONSECUTIVE_CONCURRENCY_TRIES)
        ]
        self.assertEqual(self._concurrencies, expected_concurrencies)

    def test_sweep_with_constraints(self):
        """
        Test sweeping concurrency from min to max, with 100ms latency constraint
        and throughput is linearly increasing
        """
        config = self._create_single_model_with_constraints()
        constraint_manager = ConstraintManager(config)
        concurrency_search = ConcurrencySearch(config)

        for concurrency in concurrency_search.search_concurrencies():
            self._concurrencies.append(concurrency)

            concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=100 * concurrency,
                    latency=10,
                    concurrency=concurrency,
                    constraint_manager=constraint_manager))

        expected_concurrencies = [
            2**c for c in range(self._min_concurrency_index,
                                self._max_concurrency_index + 1)
        ]

        self.assertEqual(self._concurrencies, expected_concurrencies)

    def test_not_adding_measurements(self):
        """
        Test that an exception is raised if measurements are not added
        """
        config = self._create_single_model_no_constraints()
        constraint_manager = ConstraintManager(config)
        concurrency_search = ConcurrencySearch(config)

        with self.assertRaises(TritonModelAnalyzerException):
            for concurrency in concurrency_search.search_concurrencies():
                self._concurrencies.append(concurrency)

                if concurrency < 32:
                    concurrency_search.add_run_config_measurement(
                        run_config_measurement=self._construct_rcm(
                            throughput=100 * concurrency,
                            latency=10,
                            concurrency=concurrency,
                            constraint_manager=constraint_manager))

    def _create_single_model_no_constraints(self):
        args = ['model-analyzer', 'profile', '--profile-models', 'test_model']
        yaml_str = ''
        config = evaluate_mock_config(args, yaml_str)

        return config

    def _create_single_model_with_constraints(self):
        args = [
            'model-analyzer', 'profile', '--profile-models', 'test_model',
            '--latency-budget', '100'
        ]
        yaml_str = ''
        config = evaluate_mock_config(args, yaml_str)

        return config

    def _construct_rcm(
            self, throughput: int, latency: int, concurrency: int,
            constraint_manager: ConstraintManager) -> RunConfigMeasurement:
        self.model_specific_pa_params = [{
            "batch_size": 1,
            "concurrency": concurrency
        }]

        self.rcm0_non_gpu_metric_values = [{
            # modelA_config_0
            "perf_throughput": throughput,
            "perf_latency_p99": latency,
            "cpu_used_ram": 1000
        }]

        return construct_run_config_measurement(
            model_name="test_model",
            model_config_names=["test_model_config_0"],
            model_specific_pa_params=self.model_specific_pa_params,
            gpu_metric_values=MagicMock(),
            non_gpu_metric_values=self.rcm0_non_gpu_metric_values,
            constraint_manager=constraint_manager)


if __name__ == "__main__":
    unittest.main()
