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
from unittest.mock import patch, MagicMock, Mock


class MockNVML:
    def __init__(self, test_case):
        self.test_case = test_case

    def setUp(self):
        memory_info = Mock()
        memory_info.used = 1000
        memory_info.free = 250
        memory_info.total = 123

        pci_bus_info = {
            'pci_bus_id': 101,
            'device_id': 0,
            'bus_id': 0
        }

        patcher_monitor = patch.multiple(
            'model_analyzer.monitor.nvml',
            nvmlInit=MagicMock(),
            nvmlDeviceGetMemoryInfo=Mock(return_value=memory_info),
            nvmlDeviceGetHandleByPciBusId=MagicMock()
        )

        patcher_device = patch.multiple(
            'model_analyzer.device.gpu_device_factory',
            nvmlInit=MagicMock(),
            nvmlDeviceGetHandleByPciBusId=MagicMock(),
            nvmlDeviceGetUUID=Mock(return_value='UUID'),
            nvmlDeviceGetIndex=Mock(return_value=0),
            nvmlShutdown=MagicMock()
        )

        patcher_monitor.start()
        patcher_device.start()

        self.test_case.addCleanup(patcher_monitor.stop)
        self.test_case.addCleanup(patcher_device.stop)
