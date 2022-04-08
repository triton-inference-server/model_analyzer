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

# from tests.common.test_utils import convert_non_gpu_metrics_to_data, \
#     convert_gpu_metrics_to_data, convert_avg_gpu_metrics_to_data, \
#     construct_perf_analyzer_config, construct_run_config_measurement, default_encode

# from model_analyzer.record.metrics_manager import MetricsManager
# from model_analyzer.result.model_config_measurement import ModelConfigMeasurement
# from model_analyzer.result.run_config_measurement import RunConfigMeasurement

# from statistics import mean

import unittest
from unittest.mock import MagicMock, patch
from .common import test_result_collector as trc

from model_analyzer.result.run_config_result import RunConfigResult

from tests.common.test_utils import construct_run_config_measurement, convert_non_gpu_metrics_to_data


class TestRunConfigResult(trc.TestResultCollector):

    def setUp(self):
        self._construct_empty_rcr()
        self._construct_throughput_with_latency_constraint_rcr()

    def tearDown(self):
        NotImplemented

    def test_model_name(self):
        """
        Test that model_name is correctly returned
        """
        self.assertEqual(self.rcr_empty.model_name(),
                         self.rcr_empty._model_name)

    def test_configs(self):
        """
        Test that model_configs is correctly returned
        """
        self.assertEqual(self.rcr_empty.model_configs(),
                         self.rcr_empty._model_configs)

    def test_failing_measurements_empty(self):
        """
        Test that failing returns true if no measurements are added
        """
        self.assertTrue(self.rcr_empty.failing())

    def test_failing_measurements_true(self):
        """
        Test that failing returns true if only failing measurements are present
        """
        rcr = self._rcr_throughput_with_latency_constraint

        for i in range(2, 6):
            self._add_rcm_to_rcr(rcr,
                                 throughput_value=10 * i,
                                 latency_value=100 * i)

        self.assertTrue(rcr.failing())

    def test_failing_measurements_false(self):
        """
        Test that failing returns false if any passing measurements are present
        """
        rcr = self._rcr_throughput_with_latency_constraint

        # This also tests the boundry condtion: exact match (100) is passing
        for i in range(1, 6):
            self._add_rcm_to_rcr(rcr,
                                 throughput_value=10 * i,
                                 latency_value=100 * i)

        self.assertFalse(rcr.failing())

    def test_top_n_failing(self):
        """
        Test that the top N failing measurements are returned 
        if no passing measurements are present
        """
        rcr = self._rcr_throughput_with_latency_constraint

        for i in range(2, 6):
            self._add_rcm_to_rcr(rcr,
                                 throughput_value=10 * i,
                                 latency_value=100 * i)

        # Failing measurements are returned most to least throughput
        failing_non_gpu_data = [
            convert_non_gpu_metrics_to_data({
                'perf_throughput': 10 * i,
                'perf_latency_p99': 100 * i
            }) for i in range(5, 1, -1)
        ]

        top_n_measurements = rcr.top_n_measurements(3)

        for i in range(3):
            self.assertEqual(top_n_measurements[i].non_gpu_data(),
                             [failing_non_gpu_data[i]])

    def test_top_n_passing(self):
        """
        Test that the top N passing measurements are returned 
        """

        rcr = self._rcr_throughput_with_latency_constraint

        # 4 passing, 6 failing
        for i in range(1, 11):
            self._add_rcm_to_rcr(rcr,
                                 throughput_value=10 * i,
                                 latency_value=25 * i)

        # Passing measurements are returned most to least throughput
        passing_non_gpu_data = [
            convert_non_gpu_metrics_to_data({
                'perf_throughput': 10 * i,
                'perf_latency_p99': 25 * i
            }) for i in range(4, 0, -1)
        ]

        top_n_measurements = rcr.top_n_measurements(3)

        for i in range(3):
            self.assertEqual(top_n_measurements[i].non_gpu_data(),
                             [passing_non_gpu_data[i]])

    def _construct_empty_rcr(self):
        self.rcr_empty = RunConfigResult(model_name=MagicMock(),
                                         model_configs=MagicMock(),
                                         comparator=MagicMock(),
                                         constraints=MagicMock())

    def _construct_throughput_with_latency_constraint_rcr(self):
        self._rcr_throughput_with_latency_constraint = \
            RunConfigResult(model_name=MagicMock(),
                            model_configs=MagicMock(),
                            comparator=[{'perf_throughput': 1}],
                            constraints={'perf_latency_p99': {'max': 100}})

    def _construct_single_model_rcm(self, throughput_value, latency_value):
        return construct_run_config_measurement(
            model_name='modelA',
            model_config_names=['modelA_config_0'],
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=MagicMock(),
            non_gpu_metric_values=[{
                'perf_throughput': throughput_value,
                'perf_latency_p99': latency_value
            }],
            metric_objectives=[{
                'perf_throughput': 1
            }],
            model_config_weights=[1])

    def _add_rcm_to_rcr(self, rcr, throughput_value, latency_value):
        rcr.add_run_config_measurement(
            self._construct_single_model_rcm(throughput_value, latency_value))


if __name__ == '__main__':
    unittest.main()
