# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from model_analyzer.result.run_config_result_comparator import RunConfigResultComparator
from .common import test_result_collector as trc
from .common.test_utils import construct_run_config_result

import unittest
from unittest.mock import MagicMock


class TestRunConfigResultComparatorMethods(trc.TestResultCollector):

    #TODO-MM: Add unit testing for multi-model scenarios
    def test_compare_run_config_results(self):

        # First test where throughput drives comparison
        objective_spec = {'perf_throughput': 10, 'perf_latency_p99': 5}

        avg_gpu_metrics1 = {0: {'gpu_used_memory': 5000, 'gpu_utilization': 50}}
        avg_gpu_metrics2 = {0: {'gpu_used_memory': 6000, 'gpu_utilization': 60}}

        avg_non_gpu_metrics1 = {
            'perf_throughput': 100,
            'perf_latency_p99': 4000
        }
        avg_non_gpu_metrics2 = {
            'perf_throughput': 200,
            'perf_latency_p99': 8000
        }

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            avg_gpu_metrics1=avg_gpu_metrics1,
            avg_non_gpu_metrics1=avg_non_gpu_metrics1,
            avg_gpu_metrics2=avg_gpu_metrics2,
            avg_non_gpu_metrics2=avg_non_gpu_metrics2,
            expected_result=False)

        # Latency driven
        objective_spec = {'perf_throughput': 5, 'perf_latency_p99': 10}

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            avg_gpu_metrics1=avg_gpu_metrics1,
            avg_non_gpu_metrics1=avg_non_gpu_metrics1,
            avg_gpu_metrics2=avg_gpu_metrics2,
            avg_non_gpu_metrics2=avg_non_gpu_metrics2,
            expected_result=True)

        # Equal weightage
        objective_spec = {'perf_throughput': 5, 'perf_latency_p99': 5}

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            avg_gpu_metrics1=avg_gpu_metrics1,
            avg_non_gpu_metrics1=avg_non_gpu_metrics1,
            avg_gpu_metrics2=avg_gpu_metrics2,
            avg_non_gpu_metrics2=avg_non_gpu_metrics2,
            value_step2=2,
            expected_result=False)

    def _check_run_config_result_comparison(self,
                                            objective_spec,
                                            avg_gpu_metrics1,
                                            avg_gpu_metrics2,
                                            avg_non_gpu_metrics1,
                                            avg_non_gpu_metrics2,
                                            value_step1=1,
                                            value_step2=1,
                                            expected_result=0):
        """
        Helper function that takes all the data needed to
        construct two RunConfigResults, constructs and runs a
        comparator and checks that it produces the expected
        value.
        """

        result_comparator = RunConfigResultComparator(
            metric_objectives_list=[objective_spec])

        result1 = construct_run_config_result(
            avg_gpu_metric_values=avg_gpu_metrics1,
            avg_non_gpu_metric_values=avg_non_gpu_metrics1,
            comparator=result_comparator,
            value_step=value_step1,
            model_name=MagicMock(),
            model_config=MagicMock())

        result2 = construct_run_config_result(
            avg_gpu_metric_values=avg_gpu_metrics2,
            avg_non_gpu_metric_values=avg_non_gpu_metrics2,
            comparator=result_comparator,
            value_step=value_step2,
            model_name=MagicMock(),
            model_config=MagicMock())

        self.assertEqual(result_comparator.compare(result1, result2),
                         expected_result)


# def _check_measurement_comparison(self, objective_spec, gpu_metric_values1,
#                                   non_gpu_metric_values1, gpu_metric_values2,
#                                   non_gpu_metric_values2, expected_result):
#     """
#         This function is a helper function that takes all
#         the data needed to construct two RunConfigMeasurements,
#         and constructs and runs a RunConfigResultComparator on
#         them, and checks it against an expected result
#         """

#     result_comparator = RunConfigResultComparator(
#         metric_objectives=objective_spec)

#     measurement1 = construct_run_config_measurement(MagicMock(), MagicMock(),
#                                                     MagicMock(),
#                                                     gpu_metric_values1,
#                                                     [non_gpu_metric_values1],
#                                                     [objective_spec], [1])

#     measurement2 = construct_run_config_measurement(MagicMock(), MagicMock(),
#                                                     MagicMock(),
#                                                     gpu_metric_values2,
#                                                     [non_gpu_metric_values2],
#                                                     [objective_spec], [1])

#     self.assertEqual(result_comparator.compare(measurement1, measurement2),
#                      expected_result)

if __name__ == '__main__':
    unittest.main()
