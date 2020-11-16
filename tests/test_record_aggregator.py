# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import unittest
import sys
sys.path.append('../common')

from model_analyzer.record.record_aggregator import RecordAggregator
from model_analyzer.record.perf_throughput import PerfThroughput
from model_analyzer.record.perf_latency import PerfLatency
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
import test_result_collector as trc


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
        retrieved_records = record_aggregator.filter_records()
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
            record_types=[PerfThroughput], filters=[(lambda v: v >= 5)])

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
            filters=[(lambda v: v == 3), (lambda v: v < 5)])

        retrieved_values = {
            record_type:
            [record.value() for record in retrieved_records[record_type]]
            for record_type in [PerfLatency, PerfThroughput]
        }

        self.assertEqual(len(retrieved_records[PerfLatency]), 1)
        self.assertIn(3, retrieved_values[PerfLatency])

        self.assertEqual(len(retrieved_records[PerfThroughput]), 1)
        self.assertIn(1, retrieved_values[PerfThroughput])

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
        average = lambda seq: (sum(seq) * 1.0) / len(seq)
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
