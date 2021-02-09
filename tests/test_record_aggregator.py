# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

from model_analyzer.record.record_aggregator import RecordAggregator
from model_analyzer.record.perf_throughput import PerfThroughput
from model_analyzer.record.perf_latency import PerfLatency
from .common import test_result_collector as trc


class TestRecordAggregatorMethods(trc.TestResultCollector):
    def test_insert(self):
        record_aggregator = RecordAggregator()

        self.assertEqual(record_aggregator.total(), 0)

        throughput_record = PerfThroughput("Throughput: 5 infer/sec\n\n\n\n")
        record_aggregator.insert(throughput_record)

        # Assert record is added
        self.assertEqual(record_aggregator.total(), 1)

    def test_record_types(self):
        record_aggregator = RecordAggregator()

        throughput_record = PerfThroughput("Throughput: 5 infer/sec\n\n\n\n")
        record_aggregator.insert(throughput_record)

        self.assertEqual(record_aggregator.record_types()[0], PerfThroughput)

    def test_filter_records_default(self):
        record_aggregator = RecordAggregator()

        # insert throughput record and check its presence
        throughput_record = PerfThroughput("Throughput: 5 infer/sec\n\n\n\n")
        record_aggregator.insert(throughput_record)

        # Get the record
        retrieved_records = record_aggregator.filter_records().get_records()
        retrieved_throughput = retrieved_records[PerfThroughput][0]

        self.assertIsInstance(
            retrieved_throughput,
            PerfThroughput,
            msg="Record types do not match after filter_records")

        self.assertEqual(retrieved_throughput.value(),
                         throughput_record.value(),
                         msg="Values do not match after filter_records")

    def test_filter_records_filtered(self):
        record_aggregator = RecordAggregator()

        # Test for malformed inputs
        with self.assertRaises(Exception):
            record_aggregator.filter_records(filters=[(lambda x: False)])
        with self.assertRaises(Exception):
            record_aggregator.filter_records(record_types=[None, None],
                                             filters=[(lambda x: False)])

        # Insert 3 throughputs
        record_aggregator.insert(
            PerfThroughput("Throughput: 5 infer/sec\n\n\n\n"))
        record_aggregator.insert(
            PerfThroughput("Throughput: 1 infer/sec\n\n\n\n"))
        record_aggregator.insert(
            PerfThroughput("Throughput: 10 infer/sec\n\n\n\n"))

        # Test get with filters
        retrieved_records = record_aggregator.filter_records(
            record_types=[PerfThroughput],
            filters=[(lambda v: v.value() >= 5)]).get_records()

        # Should return 2 records
        self.assertEqual(len(retrieved_records[PerfThroughput]), 2)
        retrieved_values = [
            record.value() for record in retrieved_records[PerfThroughput]
        ]
        self.assertIn(5, retrieved_values)
        self.assertIn(10, retrieved_values)

        # Insert 2 Latency records
        record_aggregator.insert(PerfLatency("Avg latency: 3 ms\n\n\n\n"))
        record_aggregator.insert(PerfLatency("Avg latency: 6 ms\n\n\n\n"))

        # Test get with multiple headers
        retrieved_records = record_aggregator.filter_records(
            record_types=[PerfLatency, PerfThroughput],
            filters=[(lambda v: v.value() == 3),
                     (lambda v: v.value() < 5)]).get_records()

        retrieved_values = {
            record_type:
            [record.value() for record in retrieved_records[record_type]]
            for record_type in [PerfLatency, PerfThroughput]
        }

        self.assertEqual(len(retrieved_records[PerfLatency]), 1)
        self.assertIn(3, retrieved_values[PerfLatency])

        self.assertEqual(len(retrieved_records[PerfThroughput]), 1)
        self.assertIn(1, retrieved_values[PerfThroughput])

    def test_groupby(self):
        record_aggregator = RecordAggregator()
        # Insert 3 throughputs
        record_aggregator.insert(
            PerfThroughput("Throughput: 5 infer/sec\n\n\n\n", timestamp=0))
        record_aggregator.insert(
            PerfThroughput("Throughput: 1 infer/sec\n\n\n\n", timestamp=1))
        record_aggregator.insert(
            PerfThroughput("Throughput: 10 infer/sec\n\n\n\n", timestamp=1))

        def groupby_criteria(record):
            return record.timestamp()

        records = record_aggregator.groupby([PerfThroughput], groupby_criteria)
        self.assertTrue(list(records[PerfThroughput]) == [0, 1])
        self.assertTrue(list(records[PerfThroughput].values()) == [5.0, 10.0])

        records = record_aggregator.groupby([PerfThroughput],
                                            groupby_criteria,
                                            reduce_func=min)
        self.assertTrue(list(records[PerfThroughput]) == [0, 1])
        self.assertTrue(list(records[PerfThroughput].values()) == [5.0, 1.0])

    def test_aggregate(self):
        record_aggregator = RecordAggregator()

        # Insert 10 records
        for i in range(10):
            record_aggregator.insert(
                PerfThroughput(f"Throughput: {i} infer/sec\n\n\n\n"))

        # Aggregate them with max, min and average
        max_vals = record_aggregator.aggregate(record_types=[PerfThroughput],
                                               reduce_func=max)
        min_vals = record_aggregator.aggregate(record_types=[PerfThroughput],
                                               reduce_func=min)

        def average(seq): return (sum(seq) * 1.0) / len(seq)
        average_vals = record_aggregator.aggregate(
            record_types=[PerfThroughput], reduce_func=average)

        self.assertEqual(max_vals[PerfThroughput],
                         9,
                         msg="Aggregation failed with max")
        self.assertEqual(min_vals[PerfThroughput],
                         0,
                         msg="Aggregation failed with min")
        self.assertEqual(average_vals[PerfThroughput],
                         4.5,
                         msg="Aggregation failed with average")


if __name__ == "__main__":
    unittest.main()
