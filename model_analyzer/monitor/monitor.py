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
from multiprocessing.pool import ThreadPool
import numba.cuda
import time

from model_analyzer.device.gpu_device_factory import GPUDeviceFactory
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class Monitor(ABC):
    """
    Monitor abstract class is a parent class used for monitoring devices.
    """
    def __init__(self, frequency, tags):
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

        # Is the background thread active
        self._thread_active = False

        # Background thread collecting results
        self._thread = None

        # Thread pool
        self._thread_pool = ThreadPool(processes=1)
        self._tags = tags

    def _monitoring_loop(self):
        self._thread_active = True
        frequency = self._frequency

        while self._thread_active:
            begin = time.time()
            # Monitoring iteration implemented by each of the subclasses
            self._monitoring_iteration(tags)

            duration = time.time() - begin
            if duration < frequency:
                time.sleep(frequency - duration)

    @abstractmethod
    def _monitoring_iteration(self):
        """
        Each of the subclasses must implement this.
        This is called to execute a single round of monitoring.
        """

        pass

    @abstractmethod
    def _collect_records(self):
        """
        This method is called to collect all the monitoring records.
        It is called in the stop_recording_metrics function after
        the background thread has stopped.
        """

        pass

    def start_recording_metrics(self):
        """
        Start recording the metrics.
        """

        self._thread = self._thread_pool.apply_async(self._monitoring_loop)

    def stop_recording_metrics(self):
        """
        Stop recording metrics. This will stop monitring all the metrics.
        """

        if not self._thread_active:
            raise TritonModelAnalyzerException(
                'start_recording_metrics should be called before\
                     stop_recording_metrics'
            )

        self._thread_active = False
        self._thread = None

        return self._collect_records()
