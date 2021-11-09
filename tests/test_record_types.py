# Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.record.record import RecordType
from .common import test_result_collector as trc


class TestRecordAggregatorMethods(trc.TestResultCollector):

    def setUp(self):
        record_types = RecordType.get_all_record_types()
        self.all_record_types = record_types.values()
        self.less_is_better_types = {
            record_types[k] for k in [
                'perf_latency_avg', 'perf_latency_p90', 'perf_latency_p95',
                'perf_latency_p99', 'gpu_used_memory', 'cpu_used_ram', 
                'perf_server_compute_infer', 'perf_latency', 
                'perf_server_queue', 'perf_client_response_wait', 
                'perf_server_compute_output', 'perf_client_send_recv', 
                'perf_server_compute_input'
            ]
        }
        self.more_is_better_types = {
            record_types[k] for k in [
                'perf_throughput', 'gpu_free_memory', 'gpu_utilization',
                'cpu_available_ram', 'gpu_power_usage'
            ]
        }

    def test_counts(self):
        """
        Make sure that all 'worse than' and 'better than' tests are tested
        """
        total_count = len(self.all_record_types)
        less_is_better_count = len(self.less_is_better_types)
        more_is_better_count = len(self.more_is_better_types)
        self.assertEqual(total_count, less_is_better_count + more_is_better_count)

    def test_add(self):
        """
        Test __add__ function for
        each record type
        """

        for record_type in self.all_record_types:
            metric1 = record_type(value=5)
            metric2 = record_type(value=9)
            metric3 = metric1 + metric2
            self.assertIsInstance(metric3, record_type)
            self.assertEqual(metric3.value(), 14)

    def test_sub(self):
        """
        Test __sub__ function for
        each record type
        """

        for record_type in self.all_record_types:
            metric1 = record_type(value=10)
            metric2 = record_type(value=3)
            metric3 = metric1 - metric2
            self.assertIsInstance(metric3, record_type)
            if record_type in self.less_is_better_types:
                self.assertEqual(metric3.value(), -7)
            elif record_type in self.more_is_better_types:
                self.assertEqual(metric3.value(), 7)
            else:
                fail

    def test_mult(self):
        """
        Test __mult__ function for
        each record type
        """

        for record_type in self.all_record_types:
            metric1 = record_type(value=6)
            metric2 = metric1 * 2
            self.assertIsInstance(metric2, record_type)
            self.assertEqual(metric2.value(), 12)

    def test_div(self):
        """
        Test __div__ function for
        each record type
        """

        for record_type in self.all_record_types:
            metric1 = record_type(value=60)
            metric2 = metric1 / 12
            self.assertIsInstance(metric2, record_type)
            self.assertEqual(metric2.value(), 5)

    def test_compare(self):
        """
        Test __lt__, __eq__, __gt__  
        functions for each record type
        """

        for record_type in self.all_record_types:
            metric1 = record_type(value=10.6)
            metric2 = record_type(value=3.2)

            # Test __lt__ (True if 1 worse than 2)
            if record_type in self.less_is_better_types:
                self.assertTrue(metric1 < metric2)
            elif record_type in self.more_is_better_types:
                self.assertTrue(metric2 < metric1)
            else:
                fail

            # Test __gt__ (True if 1 better than 2)
            if record_type in self.less_is_better_types:
                self.assertTrue(metric2 > metric1)
            elif record_type in self.more_is_better_types:
                self.assertTrue(metric1 > metric2)
            else:
                fail

            # Test __eq__
            metric1 = record_type(value=12)
            metric2 = record_type(value=12)
            self.assertEqual(metric1, metric2)


if __name__ == "__main__":
    unittest.main()
