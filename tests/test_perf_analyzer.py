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

from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.server.server_factory import TritonServerFactory
from model_analyzer.perf_analyzer.perf_analyzer import PerfAnalyzer
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.record.perf_throughput import PerfThroughput
from model_analyzer.record.perf_latency import PerfLatency

# Test Parameters
MODEL_LOCAL_PATH = '/model_analyzer/models'
MODEL_REPOSITORY_PATH = '/model_analyzer/models'
TRITON_VERSION = '20.09'
TEST_MODEL_NAME = 'classification_chestxray_v1'
CONFIG_TEST_ARG = 'sync'
TEST_RUN_PARAMS = {'batch-size': [1, 2], 'concurrency-range': [2, 4]}


class TestPerfAnalyzerMethods(unittest.TestCase):
    def setUp(self):
        # PerfAnalyzer config for all tests
        self.config = PerfAnalyzerConfig()
        self.config['model-name'] = TEST_MODEL_NAME

        # Triton Server
        self.server = None

    def test_perf_analyzer_config(self):
        # Check config initializations
        self.assertIsNone(self.config[CONFIG_TEST_ARG],
                          msg="Server config had unexpected initial"
                          f" value for {CONFIG_TEST_ARG}")
        # Set value
        self.config[CONFIG_TEST_ARG] = True

        # Test get again
        self.assertTrue(self.config[CONFIG_TEST_ARG],
                        msg=f"{CONFIG_TEST_ARG} was not set")

        # Try to set an unsupported config argument, expect failure
        with self.assertRaises(TritonModelAnalyzerException,
                               msg="Expected exception on trying to set"
                               "unsupported argument in perf_analyzer"
                               "config"):
            self.config['dummy'] = 1

    def test_run(self):
        # Now create a server config
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH

        # Create server, PerfAnalyzer, and wait for server ready
        self.server = TritonServerFactory.create_server_local(
            version=TRITON_VERSION, config=server_config)
        perf_client = PerfAnalyzer(config=self.config)

        self.server.start()
        self.server.wait_for_ready(num_retries=50)

        # Run perf analyzer
        throughput_record, latency_record = perf_client.run()

        self.server.stop()

    def test_parse_perf_output(self):
        perf_client = PerfAnalyzer(config=self.config)

        # Test latency parsing (output is at least 4 lines)
        test_latency_output = "Avg latency: 5000 ms\n\n\n\n"
        _, latency_record = perf_client._parse_perf_output(test_latency_output)
        self.assertEqual(latency_record.value(), 5000)

        # Test throughput parsing
        test_throughput_output = "Throughput: 46.8 ms\n\n\n\n"
        throughput_record, _ = perf_client._parse_perf_output(
            test_throughput_output)
        self.assertEqual(throughput_record.value(), 46.8)

        # Test parsing for both
        test_both_output = "Throughput: 0.001 ms\nAvg latency: 3.6 ms\n\n\n\n"
        throughput_record, latency_record = perf_client._parse_perf_output(
            test_both_output)
        self.assertEqual(throughput_record.value(), 0.001)
        self.assertEqual(latency_record.value(), 3.6)

    def tearDown(self):
        # In case test raises exception
        if self.server is not None:
            self.server.stop()


if __name__ == '__main__':
    unittest.main()
