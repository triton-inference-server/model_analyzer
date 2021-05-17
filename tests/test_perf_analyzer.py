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

from .mocks.mock_server_local import MockServerLocalMethods
from .mocks.mock_perf_analyzer import MockPerfAnalyzerMethods
from .mocks.mock_client import MockTritonClientMethods
from .mocks.mock_psutil import MockPSUtil

from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.server.server_factory import TritonServerFactory
from model_analyzer.triton.client.client_factory import TritonClientFactory
from model_analyzer.perf_analyzer.perf_analyzer import PerfAnalyzer
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from model_analyzer.record.types.perf_throughput import PerfThroughput
from model_analyzer.record.types.perf_latency import PerfLatency
from model_analyzer.constants import MAX_INTERVAL_CHANGES
from .common import test_result_collector as trc

# Test Parameters
MODEL_LOCAL_PATH = '/model_analyzer/models'
MODEL_REPOSITORY_PATH = '/model_analyzer/models'
PERF_BIN_PATH = 'perf_analyzer'
TRITON_LOCAL_BIN_PATH = 'test_path'
TRITON_VERSION = '21.04'
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
        self.mock_psutil = MockPSUtil()
        self.mock_psutil.start()
        self.server_local_mock.start()
        self.perf_mock.start()
        self.client_mock.start()

        # PerfAnalyzer config for all tests
        self.config = PerfAnalyzerConfig()
        self.config['model-name'] = TEST_MODEL_NAME
        self.config['measurement-interval'] = 1000

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
        perf_analyzer = PerfAnalyzer(path=PERF_BIN_PATH,
                                     config=self.config,
                                     timeout=100,
                                     max_cpu_util=50)
        self.client = TritonClientFactory.create_grpc_client(
            server_url=TEST_GRPC_URL)
        self.server.start()
        self.client.wait_for_server_ready(num_retries=1)

        # Run perf analyzer with dummy metrics to check command parsing
        perf_metrics = [id]
        test_latency_output = "Client:\n  p99 latency: 5000 us\n\n\n\n"
        self.perf_mock.set_perf_analyzer_result_string(test_latency_output)
        with self.assertRaises(TritonModelAnalyzerException):
            perf_analyzer.run(perf_metrics)

        # Test latency parsing
        test_latency_output = "Client:\n  p99 latency: 5000 us\n\n\n\n"
        self.perf_mock.set_perf_analyzer_result_string(test_latency_output)
        perf_metrics = [PerfLatency]
        perf_analyzer.run(perf_metrics)
        records = perf_analyzer.get_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].value(), 5)

        # Test throughput parsing
        test_throughput_output = "Client:\n  Throughput: 46.8 infer/sec\n\n\n\n"
        self.perf_mock.set_perf_analyzer_result_string(test_throughput_output)
        perf_metrics = [PerfThroughput]
        perf_analyzer.run(perf_metrics)
        records = perf_analyzer.get_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].value(), 46.8)

        # Test parsing for both
        test_both_output = "Client:\n  Throughput: 0.001 infer/sec\np99 latency: 3600 us\n\n\n\n"
        self.perf_mock.set_perf_analyzer_result_string(test_both_output)
        perf_metrics = [PerfThroughput, PerfLatency]
        perf_analyzer.run(perf_metrics)
        records = perf_analyzer.get_records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].value(), 0.001)
        self.assertEqual(records[1].value(), 3.6)

        # Test exception handling
        self.perf_mock.set_perf_analyzer_return_code(1)
        self.assertTrue(perf_analyzer.run(perf_metrics), 1)
        self.server.stop()

        # TODO: test perf_analyzer over utilization of resources.

    def test_measurement_interval_increase(self):
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH

        # Create server, client, PerfAnalyzer, and wait for server ready
        self.server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config)
        perf_analyzer_config = PerfAnalyzerConfig()
        perf_analyzer_config['model-name'] = TEST_MODEL_NAME
        perf_analyzer_config['concurrency-range'] = TEST_CONCURRENCY_RANGE
        perf_analyzer_config['measurement-mode'] = 'time_windows'
        perf_analyzer = PerfAnalyzer(path=PERF_BIN_PATH,
                                     config=self.config,
                                     timeout=100,
                                     max_cpu_util=50)
        self.client = TritonClientFactory.create_grpc_client(
            server_url=TEST_GRPC_URL)
        self.server.start()

        # Test failure to stabilize for measurement windows
        self.client.wait_for_server_ready(num_retries=1)
        test_stabilize_output = "Please use a larger time window"
        self.perf_mock.set_perf_analyzer_result_string(test_stabilize_output)
        self.perf_mock.set_perf_analyzer_return_code(1)
        perf_metrics = [PerfThroughput, PerfLatency]
        perf_analyzer.run(perf_metrics)
        self.assertTrue(self.perf_mock.get_perf_analyzer_popen_read_call_count(
        ) == MAX_INTERVAL_CHANGES)

    def test_measurement_request_count_increase(self):
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH

        # Create server, client, PerfAnalyzer, and wait for server ready
        self.server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config)
        perf_analyzer = PerfAnalyzer(path=PERF_BIN_PATH,
                                     config=self.config,
                                     timeout=100,
                                     max_cpu_util=50)
        self.client = TritonClientFactory.create_grpc_client(
            server_url=TEST_GRPC_URL)
        self.server.start()

        # Test the timeout for count mode
        self.client.wait_for_server_ready(num_retries=1)
        test_both_output = "Please use a larger time window"
        self.perf_mock.set_perf_analyzer_result_string(test_both_output)
        self.perf_mock.set_perf_analyzer_return_code(1)
        perf_metrics = [PerfThroughput, PerfLatency]
        perf_analyzer.run(perf_metrics)
        self.assertTrue(self.perf_mock.get_perf_analyzer_popen_read_call_count(
        ) == MAX_INTERVAL_CHANGES)

    def tearDown(self):
        # In case test raises exception
        if self.server is not None:
            self.server.stop()

        # Stop mocking
        self.server_local_mock.stop()
        self.perf_mock.stop()
        self.client_mock.stop()
        self.mock_psutil.stop()


if __name__ == '__main__':
    unittest.main()
