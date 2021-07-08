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
import time

from model_analyzer.monitor.cpu_monitor import CPUMonitor
from model_analyzer.record.types.cpu_available_ram import CPUAvailableRAM
from model_analyzer.record.types.cpu_used_ram import CPUUsedRAM
from model_analyzer.triton.server.server_factory import TritonServerFactory
from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from .common import test_result_collector as trc
from .mocks.mock_server_local import MockServerLocalMethods

MODEL_REPOSITORY_PATH = 'test_repo'
TRITON_LOCAL_BIN_PATH = 'test_bin_path/tritonserver'


class TestCPUMonitor(trc.TestResultCollector):
    def setUp(self):
        self.server_local_mock = MockServerLocalMethods()
        self.server_local_mock.start()

    def test_record_cpu_memory(self):
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH
        gpus = ['all']

        frequency = 1
        monitoring_time = 10
        metrics = [CPUAvailableRAM, CPUUsedRAM]

        server = TritonServerFactory.create_server_local(
            path=TRITON_LOCAL_BIN_PATH, config=server_config, gpus=['all'])

        # Start triton and monitor
        server.start()
        cpu_monitor = CPUMonitor(server, frequency, metrics)
        cpu_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = cpu_monitor.stop_recording_metrics()

        # Assert library calls
        self.server_local_mock.assert_cpu_stats_called()

        # Assert instance types
        for record in records:
            self.assertIsInstance(record.value(), float)
            self.assertIsInstance(record.timestamp(), int)

        # The number of records should be dividable by number of metrics
        self.assertTrue(len(records) % len(metrics) == 0)
        self.assertTrue(len(records) > 0)

        with self.assertRaises(TritonModelAnalyzerException):
            cpu_monitor.stop_recording_metrics()

        cpu_monitor.destroy()
        server.stop()

    def tearDown(self):
        self.server_local_mock.stop()


if __name__ == '__main__':
    unittest.main()
