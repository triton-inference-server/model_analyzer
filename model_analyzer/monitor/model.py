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
from abc import ABC, abstractmethod
import numba.cuda

from model_analyzer.device.gpu_device_factory import GPUDeviceFactory


class Monitor(ABC):
    """
    Monitor abstract class is a parent class used for monitoring devices.
    """

    def __init__(self, frequency):
        """
        Parameters
        ----------
        frequency : float
            How often the metrics should be monitored.
        """
        self._frequency = frequency
        self._gpus = []
        for gpu in numba.cuda.list_devices():
            gpu_device = GPUDeviceFactory.create_device_by_cuda_index(gpu.id)
            self._gpus.append(gpu_device)

    @abstractmethod
    def start_recording_metrics(self, tags):
        """
        Start recording the metrics in `tags`.

        Paramters
        ---------
        tags : list
            List of metrics to start watching for
        """
        return

    @abstractmethod
    def stop_recording_metrics(self):
        """
        Stop recording metrics. This will stop monitring all the metrics.
        """
        return
