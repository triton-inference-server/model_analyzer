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
from pynvml import nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByPciBusId,\
    nvmlInit
from multiprocessing.pool import ThreadPool
import time

from model_analyzer.monitor.model import Monitor

from model_analyzer.record.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.gpu_used_memory import GPUUsedMemory


class NVMLMonitor(Monitor):
    """
    Use NVML to monitor GPU metrics
    """
    def __init__(self, frequency):
        """
        Parameters
        ----------
        frequency : int
            Sampling frequency for the metric
        """
        super().__init__(frequency)
        nvmlInit()

        self._nvml_handles = []
        for gpu in self._gpus:
            self._nvml_handles.append(
                nvmlDeviceGetHandleByPciBusId(gpu.pci_bus_id()))

        # Is the background thread active
        self._thread_active = False

        # Background thread collecting results
        self._thread = None

        # Thread pool
        self._thread_pool = ThreadPool(processes=1)

    def _monitoring_loop(self, tags):
        self._thread_active = True
        frequency = self._frequency

        records = []
        while self._thread_active:
            begin = time.time()
            for tag in tags:
                if tag == 'memory':
                    for i, handle in enumerate(self._nvml_handles):
                        memory = nvmlDeviceGetMemoryInfo(handle)
                        gpu_device = self._gpus[i]
                        records.append(
                            GPUUsedMemory(device=gpu_device,
                                          used_mem=memory.used))
                        records.append(
                            GPUFreeMemory(device=gpu_device,
                                          free_mem=memory.free))

            duration = time.time() - begin
            if duration < frequency:
                time.sleep(frequency - duration)

        return records

    def start_recording_metrics(self, tags):
        """
        Start recording a list of tags

        Parameters
        ----------
        tags : list
            Name of the metrics to be collected. Options include
            *memory* and *compute*.
        """
        self._thread = self._thread_pool.apply_async(self._monitoring_loop,
                                                     (tags, ))

    def stop_recording_metrics(self):
        """
        Stop recording metrics

        Returns
        -------
        List of Records
            GPUUsedMemory and GPUFreeMemory in this list
        """
        self._thread_active = False
        records = self._thread.get()
        self._thread = None

        return records

    def destroy(self):
        """
        Destroy the NVMLMonitor. This function must be called
        in order to appropriately deallocate the resources.
        """
        self._thread_pool.terminate()
        self._thread_pool.close()
