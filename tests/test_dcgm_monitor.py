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

from model_analyzer.monitor.dcgm.dcgm_monitor import DCGMMonitor
from model_analyzer.record.types.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.types.gpu_power_usage import GPUPowerUsage
from model_analyzer.record.types.gpu_used_memory import GPUUsedMemory
from model_analyzer.record.types.gpu_utilization import GPUUtilization
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from .common import test_result_collector as trc
from .mocks.mock_dcgm import MockDCGM
from .mocks.mock_numba import MockNumba
from .mocks.mock_dcgm_agent import TEST_UUID
from .mocks.mock_dcgm_field_group_watcher import TEST_RECORD_VALUE


class TestDCGMMonitor(trc.TestResultCollector):
    def setUp(self):
        self.mock_dcgm = MockDCGM()
        self.mock_numba = MockNumba()
        self.mock_dcgm.start()
        self.mock_numba.start()

    def test_record_memory(self):
        # One measurement every 0.01 seconds
        frequency = 0.01
        monitoring_time = 10
        metrics = [GPUUsedMemory, GPUFreeMemory]
        gpus = ['all']
        dcgm_monitor = DCGMMonitor(gpus, frequency, metrics)
        dcgm_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = dcgm_monitor.stop_recording_metrics()

        # Assert instance types
        for record in records:
            self.assertIsInstance(record.device(), GPUDevice)
            self.assertIsInstance(record.value(), float)
            self.assertTrue(record.value() == TEST_RECORD_VALUE)
            self.assertIsInstance(record.timestamp(), int)

        # The number of records should be dividable by number of metrics
        self.assertTrue(len(records) % len(metrics) == 0)
        self.assertTrue(len(records) > 0)
        self.assertTrue(records[-1].timestamp() -
                        records[0].timestamp() >= monitoring_time)

        with self.assertRaises(TritonModelAnalyzerException):
            dcgm_monitor.stop_recording_metrics()

        dcgm_monitor.destroy()

        metrics = ['UndefinedTag']
        with self.assertRaises(TritonModelAnalyzerException):
            DCGMMonitor(gpus, frequency, metrics)

    def test_record_power(self):
        # One measurement every 0.01 seconds
        frequency = 0.01
        monitoring_time = 10
        metrics = [GPUPowerUsage]
        gpus = ['all']
        dcgm_monitor = DCGMMonitor(gpus, frequency, metrics)
        dcgm_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = dcgm_monitor.stop_recording_metrics()

        # Assert instance types
        for record in records:
            self.assertIsInstance(record.device(), GPUDevice)
            self.assertIsInstance(record.value(), float)
            self.assertTrue(record.value() == TEST_RECORD_VALUE)
            self.assertIsInstance(record.timestamp(), int)

        # The number of records should be dividable by number of metrics
        self.assertTrue(len(records) % len(metrics) == 0)
        self.assertTrue(len(records) > 0)
        self.assertTrue(records[-1].timestamp() -
                        records[0].timestamp() >= monitoring_time)

        dcgm_monitor.destroy()

    def test_record_utilization(self):
        # One measurement every 0.01 seconds
        frequency = 0.01
        monitoring_time = 10
        metrics = [GPUUtilization]
        gpus = ['all']
        dcgm_monitor = DCGMMonitor(gpus, frequency, metrics)
        dcgm_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = dcgm_monitor.stop_recording_metrics()

        # Assert instance types
        for record in records:
            self.assertIsInstance(record.device(), GPUDevice)
            self.assertIsInstance(record.value(), float)
            self.assertTrue(record.value() <= 100)
            self.assertTrue(record.value() == TEST_RECORD_VALUE)
            self.assertIsInstance(record.timestamp(), int)

        # The number of records should be dividable by number of metrics
        self.assertTrue(len(records) % len(metrics) == 0)
        self.assertTrue(len(records) > 0)
        self.assertTrue(records[-1].timestamp() -
                        records[0].timestamp() >= monitoring_time)

        dcgm_monitor.destroy()

    def test_immediate_start_stop(self):
        frequency = 0.01
        metrics = [GPUUsedMemory, GPUFreeMemory]
        gpus = ['all']
        dcgm_monitor = DCGMMonitor(gpus, frequency, metrics)
        dcgm_monitor.start_recording_metrics()
        dcgm_monitor.stop_recording_metrics()
        dcgm_monitor.destroy()

    def test_gpu_id(self):
        frequency = 0.01
        metrics = [GPUUsedMemory, GPUFreeMemory]
        gpus = ['UndefinedId']
        with self.assertRaises(TritonModelAnalyzerException):
            DCGMMonitor(gpus, frequency, metrics)

        gpus = [str(TEST_UUID, encoding='ascii')]
        dcgm_monitor = DCGMMonitor(gpus, frequency, metrics)
        dcgm_monitor.destroy()

    def tearDown(self):
        self.mock_dcgm.stop()
        self.mock_numba.stop()


if __name__ == '__main__':
    unittest.main()
