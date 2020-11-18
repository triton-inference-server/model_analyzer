# Copyright 2020, NVIDIA CORPORATION.
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
from unittest.mock import patch, MagicMock
import sys
sys.path.append('../common/')

from .mock_server_local import MockServerLocalMethods
from .mock_perf_analyzer import MockPerfAnalyzerMethods
from .mock_client import MockTritonClientMethods

from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.server.server_factory import TritonServerFactory
from model_analyzer.triton.client.client_factory import TritonClientFactory
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
TEST_GRPC_URL = 'test_hostname:test_port'


class TestPerfAnalyzerMethods(trc.TestResultCollector):
    def setUp(self):
        # Mocks
        self.server_local_mock = MockServerLocalMethods()
        self.perf_mock = MockPerfAnalyzerMethods()
        self.client_mock = MockTritonClientMethods()

        # PerfAnalyzer config for all tests
        self.config = PerfAnalyzerConfig()
        self.config['model-name'] = TEST_MODEL_NAME

        # Triton Server
        self.server = None
        self.client = None

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

    def test_run(self):
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH

        # Create server, client, PerfAnalyzer, and wait for server ready
        self.server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config)
        perf_analyzer = PerfAnalyzer(path=PERF_BIN_PATH, config=self.config)
        self.client = TritonClientFactory.create_grpc_client(
            server_url=TEST_GRPC_URL)
        self.server.start()
        self.client.wait_for_server_ready(num_retries=1)

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
        self.client_mock.stop()


if __name__ == '__main__':
    unittest.main()
