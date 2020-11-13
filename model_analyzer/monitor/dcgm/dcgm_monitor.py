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

from model_analyzer.record.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.gpu_used_memory import GPUUsedMemory
from model_analyzer.record.gpu_utilization import GPUUtilization
from model_analyzer.monitor.monitor import Monitor
from model_analyzer.model_analyzer_exceptions import \
    TritonModelAnalyzerException

import model_analyzer.monitor.dcgm.dcgm_agent as dcgm_agent
import model_analyzer.monitor.dcgm.dcgm_fields as dcgm_fields
import model_analyzer.monitor.dcgm.dcgm_field_helpers as dcgm_field_helpers
import model_analyzer.monitor.dcgm.dcgm_structs as structs

from multiprocessing.pool import ThreadPool
import time


class DCGMMonitor(Monitor):
    """
    Use DCGM to monitor GPU metrics
    """

    # Mapping between the DCGM Fields and Model Analyzer Records
    model_analyzer_to_dcgm_field = {
        GPUUsedMemory: dcgm_fields.DCGM_FI_DEV_FB_USED,
        GPUFreeMemory: dcgm_fields.DCGM_FI_DEV_FB_FREE,
        GPUUtilization: dcgm_fields.DCGM_FI_DEV_GPU_UTIL
    }

    def __init__(self, frequency, tags, dcgmPath=None):
        """
        Parameters
        ----------
        frequency : int
            Sampling frequency for the metric
        tags : list
            List of Record types to monitor
        dcgmPath : str (optional)
            DCGM installation path
        """

        super().__init__(frequency, tags)
        structs._dcgmInit(dcgmPath)
        dcgm_agent.dcgmInit()

        # Start DCGM in the embedded mode to use the shared library
        self.dcgm_handle = dcgm_handle = dcgm_agent.dcgmStartEmbedded(
            structs.DCGM_OPERATION_MODE_MANUAL)

        # Create DCGM monitor group
        self.group_id = dcgm_agent.dcgmGroupCreate(dcgm_handle,
                                                   structs.DCGM_GROUP_EMPTY,
                                                   "triton-monitor")
        # Add the GPUs to the group
        for gpu in self._gpus:
            dcgm_agent.dcgmGroupAddDevice(dcgm_handle, self.group_id,
                                          gpu.device_id())

        frequency = int(self._frequency * 1000)
        fields = []
        for tag in tags:
            if tag in self.model_analyzer_to_dcgm_field:
                dcgm_field = self.model_analyzer_to_dcgm_field[tag]
                fields.append(dcgm_field)
            else:
                dcgm_agent.dcgmShutdown()
                raise TritonModelAnalyzerException(
                    f'{tag} is not supported by Model Analyzer DCGM Monitor')

        self.dcgm_field_group_id = dcgm_agent.dcgmFieldGroupCreate(
            dcgm_handle, fields, 'triton-monitor')

        self.group_watcher = dcgm_field_helpers.DcgmFieldGroupWatcher(
            dcgm_handle, self.group_id, self.dcgm_field_group_id.value,
            structs.DCGM_OPERATION_MODE_MANUAL, frequency, 3600, 0, 0)

    def _monitoring_iteration(self):
        self.group_watcher.GetMore()

    def _collect_records(self):
        records = []
        for gpu in self._gpus:
            device_id = gpu.device_id()
            metrics = self.group_watcher.values[device_id]

            # Find the first key in the metrics dictionary to find the
            # dictionary length
            first_key = list(metrics)[0]
            num_metrics = len(metrics[first_key].values)
            for i in range(num_metrics):
                for tag in self._tags:
                    dcgm_field = self.model_analyzer_to_dcgm_field[tag]

                    # DCGM timestamp is in nanoseconds
                    records.append(
                        tag(gpu, float(metrics[dcgm_field].values[i].value),
                            metrics[dcgm_field].values[i].ts))

        return records

    def destroy(self):
        """
        Destroy the DCGMMonitor. This function must be called
        in order to appropriately deallocate the resources.
        """

        dcgm_agent.dcgmShutdown()
        self._thread_pool.terminate()
        self._thread_pool.close()
