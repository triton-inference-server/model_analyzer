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

from model_analyzer.result.result_heap import ResultHeap
from model_analyzer.result.result_comparator import ResultComparator
from .common import test_result_collector as trc
from .common.test_utils import construct_result

import unittest
from random import sample


class TestResultHeapMethods(trc.TestResultCollector):

    def setUp(self):
        objective_spec = {'perf_throughput': 10, 'perf_latency_p99': 5}
        self.result_heap = ResultHeap()
        self.result_comparator = ResultComparator(
            metric_objectives=objective_spec)

    def test_empty(self):
        avg_gpu_metrics = {0: {'gpu_used_memory': 6000, 'gpu_utilization': 60}}
        avg_non_gpu_metrics = {'perf_throughput': 100, 'perf_latency_p99': 4000}
        result = construct_result(avg_gpu_metric_values=avg_gpu_metrics,
                                  avg_non_gpu_metric_values=avg_non_gpu_metrics,
                                  comparator=self.result_comparator)
        self.assertTrue(self.result_heap.empty())
        self.result_heap.add_result(result=result)
        self.assertFalse(self.result_heap.empty())
        self.result_heap.next_best_result()
        self.assertTrue(self.result_heap.empty())

    def test_add_results(self):
        avg_gpu_metrics = {0: {'gpu_used_memory': 6000, 'gpu_utilization': 60}}
        avg_non_gpu_metrics = {'perf_throughput': 100, 'perf_latency_p99': 4000}
        for _ in range(10):
            self.result_heap.add_result(
                construct_result(avg_gpu_metric_values=avg_gpu_metrics,
                                 avg_non_gpu_metric_values=avg_non_gpu_metrics,
                                 comparator=self.result_comparator))

        results = self.result_heap.results()
        self.assertEqual(len(results), 10)

    def test_next_best_result(self):
        avg_gpu_metrics = {0: {'gpu_used_memory': 6000, 'gpu_utilization': 60}}
        for i in range(10, 0, -1):
            avg_non_gpu_metrics = {
                'perf_throughput': 100 + 10 * i,
                'perf_latency_p99': 4000
            }
            self.result_heap.add_result(
                construct_result(avg_gpu_metric_values=avg_gpu_metrics,
                                 avg_non_gpu_metric_values=avg_non_gpu_metrics,
                                 comparator=self.result_comparator,
                                 model_name=str(i)))
        self.assertEqual(self.result_heap.next_best_result().model_name(), '10')
        self.assertEqual(self.result_heap.next_best_result().model_name(), '9')
        self.assertEqual(self.result_heap.next_best_result().model_name(), '8')
        self.assertEqual(self.result_heap.next_best_result().model_name(), '7')

    def test_top_n_results(self):
        avg_gpu_metrics = {0: {'gpu_used_memory': 6000, 'gpu_utilization': 60}}
        for i in sample(range(10), 10):
            avg_non_gpu_metrics = {
                'perf_throughput': 100 + 10 * i,
                'perf_latency_p99': 4000
            }
            self.result_heap.add_result(
                construct_result(avg_gpu_metric_values=avg_gpu_metrics,
                                 avg_non_gpu_metric_values=avg_non_gpu_metrics,
                                 comparator=self.result_comparator,
                                 model_name=str(i)))

        top_5_results = self.result_heap.top_n_results(n=5)
        self.assertEqual(top_5_results[0].model_name(), '9')
        self.assertEqual(top_5_results[1].model_name(), '8')
        self.assertEqual(top_5_results[2].model_name(), '7')
        self.assertEqual(top_5_results[3].model_name(), '6')
        self.assertEqual(top_5_results[4].model_name(), '5')

        all_results = self.result_heap.top_n_results(n=-1)
        self.assertEqual(all_results[0].model_name(), '9')
        self.assertEqual(all_results[1].model_name(), '8')
        self.assertEqual(all_results[2].model_name(), '7')
        self.assertEqual(all_results[3].model_name(), '6')
        self.assertEqual(all_results[4].model_name(), '5')
        self.assertEqual(all_results[5].model_name(), '4')
        self.assertEqual(all_results[6].model_name(), '3')
        self.assertEqual(all_results[7].model_name(), '2')
        self.assertEqual(all_results[8].model_name(), '1')
        self.assertEqual(all_results[9].model_name(), '0')


if __name__ == '__main__':
    unittest.main()
