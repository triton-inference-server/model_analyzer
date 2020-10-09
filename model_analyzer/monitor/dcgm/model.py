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
from model_analyzer.record.gpu_memory_record import GPUMemoryRecord
from model_analyzer.record.record_collector import RecordCollector
from model_analyzer.monitor.model import Monitor
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

    def __init__(self, frequency, dcgmPath=None):
        """
        Parameters
        ----------
        frequency : int
            Sampling frequency for the metric
        """
        super().__init__(frequency)
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

        # Is the background thread active
        self._thread_active = False

        # Background thread collecting results
        self._thread = None

        # Thread pool
        self._thread_pool = ThreadPool(processes=1)

    def start_recording_metrics(self, tags):
        """Start recording a list of tags

        Parameters
        ----------
        tags : list
            Name of the metrics to be collected. Options include
            *memory* and *compute*.
        """
        dcgm_handle = self.dcgm_handle
        group_id = self.group_id
        frequency = int(self._frequency * 1000)

        # Metrics to watch for
        fields = []
        if 'memory' in tags:
            fields += [
                dcgm_fields.DCGM_FI_DEV_FB_FREE,
                dcgm_fields.DCGM_FI_DEV_FB_TOTAL,
                dcgm_fields.DCGM_FI_DEV_FB_USED
            ]

        self.dcgm_field_group_id = dcgm_agent.dcgmFieldGroupCreate(
            dcgm_handle, fields, 'triton-monitor')

        self.group_watcher = dcgm_field_helpers.DcgmFieldGroupWatcher(
            dcgm_handle, group_id, self.dcgm_field_group_id.value,
            structs.DCGM_OPERATION_MODE_MANUAL, frequency, 3600, 0, 0)

        self._thread = self._thread_pool.apply_async(self._monitoring_loop,
                                                     (tags,))

    def _monitoring_loop(self, tags):
        self._thread_active = True
        frequency = self._frequency

        while self._thread_active:
            begin = time.time()
            self.group_watcher.GetMore()
            duration = time.time() - begin
            if duration < frequency:
                time.sleep(frequency - duration)

        record_collector = RecordCollector()
        for gpu in self._gpus:
            device_id = gpu.device_id()
            metrics = self.group_watcher.values[device_id]
            if 'memory' in tags:
                num_metrics = len(
                    metrics[dcgm_fields.DCGM_FI_DEV_FB_FREE].values)
                for i in range(num_metrics):
                    free_memory = metrics[
                        dcgm_fields.DCGM_FI_DEV_FB_FREE].values[i]
                    used_memory = metrics[
                        dcgm_fields.DCGM_FI_DEV_FB_USED].values[i]
                    total_memory = metrics[
                        dcgm_fields.DCGM_FI_DEV_FB_TOTAL].values[i]
                    memory_record = GPUMemoryRecord(gpu, used_memory.value,
                                                    total_memory.value,
                                                    free_memory.value)
                    record_collector.insert(memory_record)
        return record_collector

    def stop_recording_metrics(self):
        """Stop recording metrics

        Returns
        -------
        RecordCollector
            A RecordCollector containing all the results
        """
        self._thread_active = False
        record_collector = self._thread.get()
        self._thread = None
        return record_collector

    def destroy(self):
        """Destroy the DCGMMonitor. This function must be called
        in order to appropriately deallocate the resources.
        """
        dcgm_agent.dcgmShutdown()
        self._thread_pool.terminate()
        self._thread_pool.close()
