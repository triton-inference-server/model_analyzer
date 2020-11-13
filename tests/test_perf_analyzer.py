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
from unittest.mock import patch, MagicMock
import sys
sys.path.append('../common/')

from .mock_server_local import MockServerLocalMethods
from .mock_perf_analyzer import MockPerfAnalyzerMethods

from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.server.server_factory import TritonServerFactory
from model_analyzer.perf_analyzer.perf_analyzer import PerfAnalyzer
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.record.perf_throughput import PerfThroughput
from model_analyzer.record.perf_latency import PerfLatency
import test_result_collector as trc

# Test Parameters
MODEL_LOCAL_PATH = '/model_analyzer/models'
MODEL_REPOSITORY_PATH = '/model_analyzer/models'
PERF_BIN_PATH = 'perf_analyzer'
TRITON_LOCAL_BIN_PATH = 'test_path'
TRITON_VERSION = '20.09'
TEST_MODEL_NAME = 'test_model'
TEST_CONCURRENCY_RANGE = '1:16:2'
CONFIG_TEST_ARG = 'sync'
TEST_RUN_PARAMS = {'batch-size': [1, 2], 'concurrency-range': [2, 4]}


class TestPerfAnalyzerMethods(trc.TestResultCollector):

    def setUp(self):
        # Mocks
        self.server_local_mock = MockServerLocalMethods()
        self.perf_mock = MockPerfAnalyzerMethods()

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

        # set and get value for each subtype of arguments
        self.config['model-name'] = TEST_MODEL_NAME
        self.assertEqual(self.config['model-name'], TEST_MODEL_NAME)

        self.config['concurrency-range'] = TEST_CONCURRENCY_RANGE
        self.assertEqual(self.config['concurrency-range'],
                         TEST_CONCURRENCY_RANGE)

        self.config['extra-verbose'] = True
        self.assertTrue(self.config['extra-verbose'])

    @patch('model_analyzer.triton.server.server.requests', get=MagicMock())
    def test_run(self, requests_mock):
        # Now create a server config
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH

        # Create server, PerfAnalyzer, and wait for server ready
        self.server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config)
        perf_analyzer = PerfAnalyzer(path=PERF_BIN_PATH, config=self.config)

        self.server.start()
        requests_mock.get.return_value.status_code = 200
        self.server.wait_for_ready(num_retries=1)

        # Run perf analyzer with dummy tags to check command parsing
        perf_tags = [id]
        _ = perf_analyzer.run(perf_tags)
        self.perf_mock.assert_perf_analyzer_run_as(
            [PERF_BIN_PATH, '-m', TEST_MODEL_NAME])

        # Test latency parsing
        test_latency_output = "Avg latency: 5000 ms\n\n\n\n"
        self.perf_mock.set_perf_analyzer_result_string(test_latency_output)
        perf_tags = [PerfLatency]
        records = perf_analyzer.run(perf_tags)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].value(), 5000)

        # Test throughput parsing
        test_throughput_output = "Throughput: 46.8 infer/sec\n\n\n\n"
        self.perf_mock.set_perf_analyzer_result_string(test_throughput_output)
        perf_tags = [PerfThroughput]
        records = perf_analyzer.run(perf_tags)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].value(), 46.8)

        # Test parsing for both
        test_both_output = "Throughput: 0.001 infer/sec\nAvg latency: 3.6 ms\n\n\n\n"
        self.perf_mock.set_perf_analyzer_result_string(test_both_output)
        perf_tags = [PerfLatency, PerfThroughput]
        records = perf_analyzer.run(perf_tags)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].value(), 3.6)
        self.assertEqual(records[1].value(), 0.001)

        # Test exception handling
        with self.assertRaisesRegex(
                expected_exception=TritonModelAnalyzerException,
                expected_regex="Running perf_analyzer with",
                msg="Expected TritonModelAnalyzerException"):
            self.perf_mock.raise_exception_on_run()
            _ = perf_analyzer.run(perf_tags)

        self.server.stop()

    def tearDown(self):
        # In case test raises exception
        if self.server is not None:
            self.server.stop()

        # Stop mocking
        self.server_local_mock.stop()
        self.perf_mock.stop()


if __name__ == '__main__':
    unittest.main()
