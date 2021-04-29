# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from model_analyzer.result.result_comparator import ResultComparator
from .common import test_result_collector as trc
from .common.test_utils import construct_measurement, construct_result

import unittest


class TestResultComparatorMethods(trc.TestResultCollector):
    def _check_measurement_comparison(self, objective_spec, gpu_metric_values1,
                                      non_gpu_metric_values1,
                                      gpu_metric_values2,
                                      non_gpu_metric_values2, expected_result):
        """
        This function is a helper function that takes all 
        the data needed to construct two measurements, 
        and constructs and runs a result comparator on
        them, and checks it against an expected result
        """

        result_comparator = ResultComparator(metric_objectives=objective_spec)

        measurement1 = construct_measurement('test_model', gpu_metric_values1,
                                             non_gpu_metric_values1,
                                             result_comparator)
        measurement2 = construct_measurement('test_model', gpu_metric_values2,
                                             non_gpu_metric_values2,
                                             result_comparator)
        self.assertEqual(
            result_comparator.compare_measurements(measurement1, measurement2),
            expected_result)

    def test_compare_measurements(self):
        # First test where throughput drives comparison
        objective_spec = {'perf_throughput': 10, 'perf_latency': 5}

        gpu_metric_values1 = {
            0: {
                'gpu_used_memory': 5000,
                'gpu_utilization': 50
            }
        }
        gpu_metric_values2 = {
            0: {
                'gpu_used_memory': 6000,
                'gpu_utilization': 60
            }
        }

        non_gpu_metric_values1 = {'perf_throughput': 100, 'perf_latency': 4000}
        non_gpu_metric_values2 = {'perf_throughput': 200, 'perf_latency': 8000}

        self._check_measurement_comparison(
            objective_spec=objective_spec,
            gpu_metric_values1=gpu_metric_values1,
            non_gpu_metric_values1=non_gpu_metric_values1,
            gpu_metric_values2=gpu_metric_values2,
            non_gpu_metric_values2=non_gpu_metric_values2,
            expected_result=-1)

        # Second test where latency drives comparison
        objective_spec = {'perf_throughput': 5, 'perf_latency': 10}

        self._check_measurement_comparison(
            objective_spec=objective_spec,
            gpu_metric_values1=gpu_metric_values1,
            non_gpu_metric_values1=non_gpu_metric_values1,
            gpu_metric_values2=gpu_metric_values2,
            non_gpu_metric_values2=non_gpu_metric_values2,
            expected_result=1)

        # Third test says apply equal weightage to latency and throughput
        objective_spec = {'perf_throughput': 10, 'perf_latency': 10}

        self._check_measurement_comparison(
            objective_spec=objective_spec,
            gpu_metric_values1=gpu_metric_values1,
            non_gpu_metric_values1=non_gpu_metric_values1,
            gpu_metric_values2=gpu_metric_values2,
            non_gpu_metric_values2=non_gpu_metric_values2,
            expected_result=0)

        # Change the second set of values so that throughput is way better but
        # latency is not so much worse, and run with equal weightage objective spec
        # Expect measurement 2 > measurement 1 now
        non_gpu_metric_values2 = {'perf_throughput': 200, 'perf_latency': 6000}

        self._check_measurement_comparison(
            objective_spec=objective_spec,
            gpu_metric_values1=gpu_metric_values1,
            non_gpu_metric_values1=non_gpu_metric_values1,
            gpu_metric_values2=gpu_metric_values2,
            non_gpu_metric_values2=non_gpu_metric_values2,
            expected_result=-1)

        # Improve throughput in first set of values to be 75% second case
        non_gpu_metric_values1 = {'perf_throughput': 150, 'perf_latency': 4000}

        self._check_measurement_comparison(
            objective_spec=objective_spec,
            gpu_metric_values1=gpu_metric_values1,
            non_gpu_metric_values1=non_gpu_metric_values1,
            gpu_metric_values2=gpu_metric_values2,
            non_gpu_metric_values2=non_gpu_metric_values2,
            expected_result=1)

    def _check_result_comparison(self,
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
        construct two results, constructs and runs a result
        comparator and checks that it produces the expected
        value.
        """

        result_comparator = ResultComparator(metric_objectives=objective_spec)

        result1 = construct_result(avg_gpu_metrics1, avg_non_gpu_metrics1,
                                   result_comparator, value_step1)
        result2 = construct_result(avg_gpu_metrics2, avg_non_gpu_metrics2,
                                   result_comparator, value_step2)
        self.assertEqual(result_comparator.compare_results(result1, result2),
                         expected_result)

    def test_compare_results(self):

        # First test where throughput drives comparison
        objective_spec = {'perf_throughput': 10, 'perf_latency': 5}

        avg_gpu_metrics1 = {
            0: {
                'gpu_used_memory': 5000,
                'gpu_utilization': 50
            }
        }
        avg_gpu_metrics2 = {
            0: {
                'gpu_used_memory': 6000,
                'gpu_utilization': 60
            }
        }
        avg_non_gpu_metrics1 = {'perf_throughput': 100, 'perf_latency': 4000}
        avg_non_gpu_metrics2 = {'perf_throughput': 200, 'perf_latency': 8000}

        self._check_result_comparison(
            objective_spec=objective_spec,
            avg_gpu_metrics1=avg_gpu_metrics1,
            avg_non_gpu_metrics1=avg_non_gpu_metrics1,
            avg_gpu_metrics2=avg_gpu_metrics2,
            avg_non_gpu_metrics2=avg_non_gpu_metrics2,
            expected_result=-1)

        # Latency driven
        objective_spec = {'perf_throughput': 5, 'perf_latency': 10}

        self._check_result_comparison(
            objective_spec=objective_spec,
            avg_gpu_metrics1=avg_gpu_metrics1,
            avg_non_gpu_metrics1=avg_non_gpu_metrics1,
            avg_gpu_metrics2=avg_gpu_metrics2,
            avg_non_gpu_metrics2=avg_non_gpu_metrics2,
            expected_result=1)

        # Equal weightage
        objective_spec = {'perf_throughput': 5, 'perf_latency': 5}

        self._check_result_comparison(
            objective_spec=objective_spec,
            avg_gpu_metrics1=avg_gpu_metrics1,
            avg_non_gpu_metrics1=avg_non_gpu_metrics1,
            avg_gpu_metrics2=avg_gpu_metrics2,
            avg_non_gpu_metrics2=avg_non_gpu_metrics2,
            value_step2=2,
            expected_result=0)


if __name__ == '__main__':
    unittest.main()
