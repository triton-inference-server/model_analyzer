# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from unittest import result
from model_analyzer.record.measurement import Measurement
from model_analyzer.result.run_result import RunResult
from model_analyzer.result.result_comparator import ResultComparator
from model_analyzer.record.metrics_mapper import MetricsMapper
from .common import test_result_collector as trc


class TestResultComparatorMethods(trc.TestResultCollector):
    def _construct_result_comparator(self, gpu_metric_tags,
                                     non_gpu_metric_tags, objective_spec):
        """
        Constructs a result comparator from the given
        objective spec dictionary
        """

        gpu_metric_types = MetricsMapper.get_metric_types(gpu_metric_tags)
        non_gpu_metric_types = MetricsMapper.get_metric_types(
            non_gpu_metric_tags)
        objective_tags = list(objective_spec.keys())
        objective_metrics = MetricsMapper.get_metric_types(objective_tags)
        objectives = {
            objective_metrics[i]: objective_spec[objective_tags[i]]
            for i in range(len(objective_tags))
        }

        return ResultComparator(gpu_metric_types=gpu_metric_types,
                                non_gpu_metric_types=non_gpu_metric_types,
                                metric_objectives=objectives)

    def _construct_measurement(self, gpu_metric_values, non_gpu_metric_values,
                               comparator):
        """
        Construct a measurement from the given data
        """

        # gpu_data will be a dict whose keys are gpu_ids and values
        # are lists of Records
        gpu_data = {}
        for gpu_id, metrics_values in gpu_metric_values.items():
            gpu_data[gpu_id] = []
            gpu_metric_tags = list(metrics_values.keys())
            for i, gpu_metric in enumerate(
                    MetricsMapper.get_metric_types(gpu_metric_tags)):
                gpu_data[gpu_id].append(
                    gpu_metric(value=metrics_values[gpu_metric_tags[i]]))

        # Non gpu data will be a list of records
        non_gpu_data = []
        non_gpu_metric_tags = list(non_gpu_metric_values.keys())
        for i, metric in enumerate(
                MetricsMapper.get_metric_types(non_gpu_metric_tags)):
            non_gpu_data.append(
                metric(value=non_gpu_metric_values[non_gpu_metric_tags[i]]))

        return Measurement(gpu_data=gpu_data,
                           non_gpu_data=non_gpu_data,
                           perf_config=None,
                           comparator=comparator)

    def _check_measurement_comparison(self, gpu_metric_tags,
                                      non_gpu_metric_tags, objective_spec,
                                      gpu_metric_values1,
                                      non_gpu_metric_values1,
                                      gpu_metric_values2,
                                      non_gpu_metric_values2, expected_result):
        """
        This function is a helper function that takes all 
        the data needed to construct two measurements, 
        and constructs and runs a result comparator on
        them, and checks it against an expected result
        """

        result_comparator = self._construct_result_comparator(
            gpu_metric_tags, non_gpu_metric_tags, objective_spec)

        measurement1 = self._construct_measurement(gpu_metric_values1,
                                                   non_gpu_metric_values1,
                                                   result_comparator)
        measurement2 = self._construct_measurement(gpu_metric_values2,
                                                   non_gpu_metric_values2,
                                                   result_comparator)
        self.assertEqual(
            result_comparator.compare_measurements(measurement1, measurement2),
            expected_result)

    def test_compare_measurements(self):

        gpu_metric_tags = ['gpu_used_memory', 'gpu_utilization']
        non_gpu_metric_tags = ['perf_throughput', 'perf_latency']

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
            gpu_metric_tags=gpu_metric_tags,
            non_gpu_metric_tags=non_gpu_metric_tags,
            objective_spec=objective_spec,
            gpu_metric_values1=gpu_metric_values1,
            non_gpu_metric_values1=non_gpu_metric_values1,
            gpu_metric_values2=gpu_metric_values2,
            non_gpu_metric_values2=non_gpu_metric_values2,
            expected_result=-1)

        # Second test where latency drives comparison
        objective_spec = {'perf_throughput': 5, 'perf_latency': 10}

        self._check_measurement_comparison(
            gpu_metric_tags=gpu_metric_tags,
            non_gpu_metric_tags=non_gpu_metric_tags,
            objective_spec=objective_spec,
            gpu_metric_values1=gpu_metric_values1,
            non_gpu_metric_values1=non_gpu_metric_values1,
            gpu_metric_values2=gpu_metric_values2,
            non_gpu_metric_values2=non_gpu_metric_values2,
            expected_result=1)

        # Third test says apply equal weightage to latency and throughput
        objective_spec = {'perf_throughput': 10, 'perf_latency': 10}

        self._check_measurement_comparison(
            gpu_metric_tags=gpu_metric_tags,
            non_gpu_metric_tags=non_gpu_metric_tags,
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
            gpu_metric_tags=gpu_metric_tags,
            non_gpu_metric_tags=non_gpu_metric_tags,
            objective_spec=objective_spec,
            gpu_metric_values1=gpu_metric_values1,
            non_gpu_metric_values1=non_gpu_metric_values1,
            gpu_metric_values2=gpu_metric_values2,
            non_gpu_metric_values2=non_gpu_metric_values2,
            expected_result=-1)

        # Improve throughput in first set of values to be 75% second case
        non_gpu_metric_values1 = {'perf_throughput': 150, 'perf_latency': 4000}

        self._check_measurement_comparison(
            gpu_metric_tags=gpu_metric_tags,
            non_gpu_metric_tags=non_gpu_metric_tags,
            objective_spec=objective_spec,
            gpu_metric_values1=gpu_metric_values1,
            non_gpu_metric_values1=non_gpu_metric_values1,
            gpu_metric_values2=gpu_metric_values2,
            non_gpu_metric_values2=non_gpu_metric_values2,
            expected_result=1)

    def _construct_result(self, avg_gpu_metric_values,
                          avg_non_gpu_metric_values, comparator):
        """
        Takes a dictionary whose keys are average
        metric values, constructs artificial data 
        around these averages, and then constructs
        a result from this data.
        """

        num_vals = 10

        # Construct a result
        run_result = RunResult(run_config=None, comparator=comparator)

        # Get dict of list of metric values
        gpu_metric_values = {}
        for gpu_id, metric_values in avg_gpu_metric_values.items():
            gpu_metric_values[gpu_id] = {
                key: list(range(val - num_vals, val + num_vals))
                for key, val in metric_values.items()
            }

        non_gpu_metric_values = {
            key: list(range(val - num_vals, val + num_vals))
            for key, val in avg_non_gpu_metric_values.items()
        }

        # Construct measurements and add them to the result
        for i in range(2 * num_vals):
            gpu_metrics = {}
            for gpu_id, metric_values in gpu_metric_values.items():
                gpu_metrics[gpu_id] = {
                    key: metric_values[key][i]
                    for key in metric_values
                }
            non_gpu_metrics = {
                key: non_gpu_metric_values[key][i]
                for key in non_gpu_metric_values
            }
            run_result.add_data(
                self._construct_measurement(
                    gpu_metric_values=gpu_metrics,
                    non_gpu_metric_values=non_gpu_metrics,
                    comparator=comparator))

        return run_result

    def _check_result_comparison(self, gpu_metric_tags, non_gpu_metric_tags,
                                 objective_spec, avg_gpu_metrics1,
                                 avg_gpu_metrics2, avg_non_gpu_metrics1,
                                 avg_non_gpu_metrics2, expected_result):
        """
        Helper function that takes all the data needed to
        construct two results, constructs and runs a result
        comparator and checks that it produces the expected
        value.
        """

        result_comparator = self._construct_result_comparator(
            gpu_metric_tags, non_gpu_metric_tags, objective_spec)

        result1 = self._construct_result(avg_gpu_metrics1,
                                         avg_non_gpu_metrics1,
                                         result_comparator)
        result2 = self._construct_result(avg_gpu_metrics2,
                                         avg_non_gpu_metrics2,
                                         result_comparator)

        self.assertEqual(result_comparator.compare_results(result1, result2),
                         expected_result)

    def test_compare_results(self):
        gpu_metric_tags = ['gpu_used_memory', 'gpu_utilization']
        non_gpu_metric_tags = ['perf_throughput', 'perf_latency']

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
            gpu_metric_tags=gpu_metric_tags,
            non_gpu_metric_tags=non_gpu_metric_tags,
            objective_spec=objective_spec,
            avg_gpu_metrics1=avg_gpu_metrics1,
            avg_non_gpu_metrics1=avg_non_gpu_metrics1,
            avg_gpu_metrics2=avg_gpu_metrics2,
            avg_non_gpu_metrics2=avg_non_gpu_metrics2,
            expected_result=-1)

        # Latency driven
        objective_spec = {'perf_throughput': 5, 'perf_latency': 10}

        self._check_result_comparison(
            gpu_metric_tags=gpu_metric_tags,
            non_gpu_metric_tags=non_gpu_metric_tags,
            objective_spec=objective_spec,
            avg_gpu_metrics1=avg_gpu_metrics1,
            avg_non_gpu_metrics1=avg_non_gpu_metrics1,
            avg_gpu_metrics2=avg_gpu_metrics2,
            avg_non_gpu_metrics2=avg_non_gpu_metrics2,
            expected_result=1)

        # Equal weightage
        objective_spec = {'perf_throughput': 5, 'perf_latency': 5}

        self._check_result_comparison(
            gpu_metric_tags=gpu_metric_tags,
            non_gpu_metric_tags=non_gpu_metric_tags,
            objective_spec=objective_spec,
            avg_gpu_metrics1=avg_gpu_metrics1,
            avg_non_gpu_metrics1=avg_non_gpu_metrics1,
            avg_gpu_metrics2=avg_gpu_metrics2,
            avg_non_gpu_metrics2=avg_non_gpu_metrics2,
            expected_result=0)
