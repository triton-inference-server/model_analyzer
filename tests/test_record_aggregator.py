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

        throughput_record = PerfThroughput(5)
        record_aggregator.insert(throughput_record)

        # Assert record is added
        self.assertEqual(record_aggregator.total(), 1)

    def test_headers(self):
        record_aggregator = RecordAggregator()

        throughput_record = PerfThroughput(5)
        record_aggregator.insert(throughput_record)

        self.assertEqual(record_aggregator.headers()[0],
                         throughput_record.header())

    def test_filter_records_default(self):
        record_aggregator = RecordAggregator()

        # insert throughput record and check its presence
        throughput_record = PerfThroughput(5)
        record_aggregator.insert(throughput_record)

        # Get the record
        retrieved_records = record_aggregator.filter_records()
        retrieved_throughput = retrieved_records[throughput_record.header()][0]

        self.assertEqual(retrieved_throughput.header(),
                         throughput_record.header(),
                         msg="Headers do not match after filter_records")

        self.assertEqual(retrieved_throughput.value(),
                         throughput_record.value(),
                         msg="Values do not match after filter_records")

    def test_filter_records_filtered(self):
        record_aggregator = RecordAggregator()

        # Test for malformed inputs
        with self.assertRaises(Exception):
            record_aggregator.filter_records(filters=[(lambda x: False)])
        with self.assertRaises(Exception):
            record_aggregator.filter_records(headers=["header1", "header2"],
                                             filters=[(lambda x: False)])

        # Insert 3 throughputs
        throughput_record = PerfThroughput(5)
        record_aggregator.insert(throughput_record)
        record_aggregator.insert(PerfThroughput(1))
        record_aggregator.insert(PerfThroughput(10))

        # Test get with filters
        retrieved_records = record_aggregator.filter_records(
            headers=[throughput_record.header()], filters=[(lambda v: v >= 5)])

        # Should return 2 records
        self.assertEqual(len(retrieved_records[throughput_record.header()]), 2)

        # Insert 2 Latency records
        latency_record = PerfLatency(3)
        record_aggregator.insert(latency_record)
        record_aggregator.insert(PerfLatency(6))

        # Test get with multiple headers
        retrieved_records = record_aggregator.filter_records(
            headers=[latency_record.header(),
                     throughput_record.header()],
            filters=[(lambda v: v == 3), (lambda v: v < 5)])

        self.assertEqual(len(retrieved_records[throughput_record.header()]), 1)
        self.assertEqual(len(retrieved_records[latency_record.header()]), 1)

    def test_aggregate(self):
        record_aggregator = RecordAggregator()
        throughput_record = PerfThroughput(10)
        record_aggregator.insert(throughput_record)

        # Insert 10 records
        for i in range(9):
            record_aggregator.insert(PerfThroughput(i))

        # Aggregate them with max, min and average
        max_vals = record_aggregator.aggregate(
            headers=[throughput_record.header()], reduce_func=max)
        min_vals = record_aggregator.aggregate(
            headers=[throughput_record.header()], reduce_func=min)
        average = lambda seq: (sum(seq) * 1.0) / len(seq)
        average_vals = record_aggregator.aggregate(
            headers=[throughput_record.header()], reduce_func=average)

        self.assertEqual(max_vals[throughput_record.header()],
                         10,
                         msg="Aggregation failed with max")
        self.assertEqual(min_vals[throughput_record.header()],
                         0,
                         msg="Aggregation failed with min")
        self.assertEqual(average_vals[throughput_record.header()],
                         4.6,
                         msg="Aggregation failed with average")


if __name__ == "__main__":
    unittest.main()
