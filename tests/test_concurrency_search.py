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

from .common.test_utils import construct_run_config_measurement, evaluate_mock_config

from model_analyzer.constants import THROUGHPUT_MINIMUM_CONSECUTIVE_CONCURRENCY_TRIES

from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from model_analyzer.result.concurrency_search import ConcurrencySearch

from math import log2

import unittest
from unittest.mock import MagicMock, patch
from .common import test_result_collector as trc


class TestConcurrencySearch(trc.TestResultCollector):

    def setUp(self):
        args = ['model-analyzer', 'profile', '--profile-models', 'modelA']
        yaml_str = ''
        config = evaluate_mock_config(args, yaml_str)

        self._min_concurrency_index = int(
            log2(config.run_config_search_min_concurrency))
        self._max_concurrency_index = int(
            log2(config.run_config_search_max_concurrency))

        self._concurrency_search = ConcurrencySearch(config)
        self._concurrencies = []

    def tearDown(self):
        patch.stopall()

    def test_basic_sweep(self):
        """
        Test sweeping concurrency from min to max, when no constraints are present
        and throughput is increasing
        """
        for concurrency in self._concurrency_search.search_concurrencies():
            self._concurrencies.append(concurrency)

            self._concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=100 * concurrency,
                    latency=10,
                    concurrency=concurrency))

        expected_concurrencies = [
            2**c for c in range(self._min_concurrency_index,
                                self._max_concurrency_index + 1)
        ]

        self.assertEqual(self._concurrencies, expected_concurrencies)

    def test_basic_saturating_sweep(self):
        """
        Test sweeping concurrency from min to max, when no constraints are present
        and throughput increases and then saturates
        """

        # [100, 200, 400, 800, 1000, 1000,...]
        throughputs = [
            100 * 2**c if c < 4 else 1000 for c in range(
                self._min_concurrency_index, self._max_concurrency_index + 1)
        ]

        for i, concurrency in enumerate(
                self._concurrency_search.search_concurrencies()):
            self._concurrencies.append(concurrency)

            self._concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=throughputs[i],
                    latency=10,
                    concurrency=concurrency))

        expected_concurrencies = [
            2**c
            for c in range(4 + THROUGHPUT_MINIMUM_CONSECUTIVE_CONCURRENCY_TRIES)
        ]
        self.assertEqual(self._concurrencies, expected_concurrencies)

    def _construct_rcm(self, throughput: int, latency: int,
                       concurrency: int) -> RunConfigMeasurement:
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
            constraint_manager=MagicMock())


if __name__ == "__main__":
    unittest.main()
