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
from collections import defaultdict
import time

from model_analyzer.monitor.monitor import Monitor
from model_analyzer.record.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.gpu_used_memory import GPUUsedMemory
from model_analyzer.model_analyzer_exceptions import \
    TritonModelAnalyzerException


class NVMLMonitor(Monitor):
    """
    Use NVML to monitor GPU metrics
    """

    # This is a dictionary mapping NVML functions to Model Analyzer records.
    # Each NVML function is a key in this dictionary mapping to another
    # dictionary. In the second dictionary, the keys are Model Analyzer record
    # types and the values are the field from the return type that should be
    # accessed to retrieve this record
    MODEL_ANALYZER_TO_NVML_RECORDS = {
        nvmlDeviceGetMemoryInfo: {
            GPUFreeMemory: 'free',
            GPUUsedMemory: 'used'
        }
    }

    def __init__(self, frequency, tags):
        """
        Parameters
        ----------
        frequency : int
            Sampling frequency for the metric
        tags : list
            List of Record types to monitor
        """

        super().__init__(frequency, tags)
        nvmlInit()

        self._nvml_handles = []
        for gpu in self._gpus:
            self._nvml_handles.append(
                nvmlDeviceGetHandleByPciBusId(gpu.pci_bus_id()))

        self._field_monitors = field_monitors = defaultdict(list)
        for tag in tags:
            for nvml_function, record_types in self.MODEL_ANALYZER_TO_NVML_RECORDS.items():
                if tag in list(record_types):
                    self._field_monitors[nvml_function].append(tag)
                    break
            else:
                raise TritonModelAnalyzerException(
                    f'{tag} is not supported by Model Analyzer NVML Monitor')

        self._records = []

    def _monitoring_iteration(self):
        # Loop on all of the GPUs
        for i in range(len(self._gpus)):
            handle = self._nvml_handles[i]
            # Loop on all of the monitor functions
            for monitor_function, monitor_tags in self._field_monitors.items():
                nvml_record = monitor_function(handle)
                # Loop on every tag for this monitor function
                for monitor_tag in monitor_tags:
                    field_name = self.MODEL_ANALYZER_TO_NVML_RECORDS[
                        monitor_function][monitor_tag]
                    if not hasattr(nvml_record, field_name):
                        raise TritonModelAnalyzerException(
                            f'{monitor_tag} not found in the NVML record')

                    # Retrieve monitoring value from the NVML record
                    record_value = getattr(nvml_record, field_name)
                    # Create Model Analyzer Record
                    monitor_record = monitor_tag(self._gpus[i], record_value,
                                                 time.time())
                    self._records.append(monitor_record)

    def _collect_records(self):
        records = self._records
        # Empty self._records for future data collections
        self._records = []
        return records

    def destroy(self):
        """
        Destroy the NVMLMonitor. This function must be called
        in order to appropriately deallocate the resources.
        """

        self._thread_pool.terminate()
        self._thread_pool.close()
