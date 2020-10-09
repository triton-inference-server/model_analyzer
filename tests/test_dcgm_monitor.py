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
from model_analyzer.record.gpu_memory_record import GPUMemoryRecord
from model_analyzer.device.gpu_device import GPUDevice


class TestDCGMMonitor(unittest.TestCase):

    def test_record_memory(self):
        from model_analyzer.monitor.dcgm.model import DCGMMonitor

        # One measurement every 0.01 seconds
        frequency = 0.01
        monitoring_time = 10
        dcgm_monitor = DCGMMonitor(frequency)
        dcgm_monitor.start_recording_metrics(['memory'])
        time.sleep(monitoring_time)
        metrics = dcgm_monitor.stop_recording_metrics()
        dcgm_monitor.destroy()

        # Assert instance types
        for i in range(metrics.size()):
            metric = metrics.get(i)
            self.assertIsInstance(metric, GPUMemoryRecord)
            self.assertIsInstance(metric.device, GPUDevice)


if __name__ == '__main__':
    unittest.main()
