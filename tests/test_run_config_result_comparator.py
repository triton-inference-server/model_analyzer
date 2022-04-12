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

    def setUp(self):
        self.avg_gpu_metrics1 = {
            0: {
                'gpu_used_memory': 5000,
                'gpu_utilization': 50
            }
        }
        self.avg_gpu_metrics2 = {
            0: {
                'gpu_used_memory': 6000,
                'gpu_utilization': 60
            }
        }

        self.avg_non_gpu_metrics1 = [{
            'perf_throughput': 100,
            'perf_latency_p99': 4000
        }]

        self.avg_non_gpu_metrics2 = [{
            'perf_throughput': 200,
            'perf_latency_p99': 8000
        }]

        self.avg_non_gpu_metrics_multi1 = [{
            'perf_throughput': 100,
            'perf_latency_p99': 4000
        }, {
            'perf_throughput': 200,
            'perf_latency_p99': 5000
        }]

        self.avg_non_gpu_metrics_multi2 = [{
            'perf_throughput': 300,
            'perf_latency_p99': 6000
        }, {
            'perf_throughput': 400,
            'perf_latency_p99': 7000
        }]

    def tearDown(self):
        NotImplemented

    def test_throughput_driven(self):
        objective_spec = [{'perf_throughput': 10, 'perf_latency_p99': 5}]

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            avg_gpu_metrics1=self.avg_gpu_metrics1,
            avg_non_gpu_metrics1=self.avg_non_gpu_metrics1,
            avg_gpu_metrics2=self.avg_gpu_metrics2,
            avg_non_gpu_metrics2=self.avg_non_gpu_metrics2,
            expected_result=False)

    # def test_throughput_driven_multi_model(self):
    #     objective_spec = [{'perf_throughput': 10, 'perf_latency_p99': 5},
    #                       {'perf_throughput': 10, 'perf_latency_p99': 5}]

    def test_latency_driven(self):
        objective_spec = [{'perf_throughput': 5, 'perf_latency_p99': 10}]

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            avg_gpu_metrics1=self.avg_gpu_metrics1,
            avg_non_gpu_metrics1=self.avg_non_gpu_metrics1,
            avg_gpu_metrics2=self.avg_gpu_metrics2,
            avg_non_gpu_metrics2=self.avg_non_gpu_metrics2,
            expected_result=True)

    def test_equal_weight(self):
        objective_spec = [{'perf_throughput': 5, 'perf_latency_p99': 5}]

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            avg_gpu_metrics1=self.avg_gpu_metrics1,
            avg_non_gpu_metrics1=self.avg_non_gpu_metrics1,
            avg_gpu_metrics2=self.avg_gpu_metrics2,
            avg_non_gpu_metrics2=self.avg_non_gpu_metrics2,
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
            metric_objectives_list=objective_spec)

        result1 = construct_run_config_result(
            avg_gpu_metric_values=avg_gpu_metrics1,
            avg_non_gpu_metric_values_list=avg_non_gpu_metrics1,
            comparator=result_comparator,
            value_step=value_step1,
            run_config=MagicMock())

        result2 = construct_run_config_result(
            avg_gpu_metric_values=avg_gpu_metrics2,
            avg_non_gpu_metric_values_list=avg_non_gpu_metrics2,
            comparator=result_comparator,
            value_step=value_step2,
            run_config=MagicMock())

        self.assertEqual(result_comparator.is_better_than(result1, result2),
                         expected_result)


if __name__ == '__main__':
    unittest.main()
