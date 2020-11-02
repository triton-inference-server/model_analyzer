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
import time
import sys

from model_analyzer.monitor.nvml import NVMLMonitor
from model_analyzer.record.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.gpu_used_memory import GPUUsedMemory
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class TestNVMLMonitor(unittest.TestCase):

    def test_record_memory(self):
        # One measurement every 0.01 seconds
        frequency = 0.01
        monitoring_time = 10
        tags = [GPUUsedMemory, GPUFreeMemory]
        nvml_monitor = NVMLMonitor(frequency, tags)
        nvml_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = nvml_monitor.stop_recording_metrics()

        # Assert instance types
        for record in records:
            self.assertIsInstance(record.device(), GPUDevice)

        # The number of records should be dividable by number of tags
        self.assertTrue(len(records) % len(tags) == 0)
        self.assertTrue(len(records) > 0)

        with self.assertRaises(TritonModelAnalyzerException):
            nvml_monitor.stop_recording_metrics()

        nvml_monitor.destroy()

        tags = ['UndefinedTag']
        with self.assertRaises(TritonModelAnalyzerException):
            NVMLMonitor(frequency, tags)


if __name__ == '__main__':
    unittest.main()
