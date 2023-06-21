# Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .monitor import Monitor

from model_analyzer.record.types.cpu_available_ram import CPUAvailableRAM
from model_analyzer.record.types.cpu_used_ram import CPUUsedRAM


class CPUMonitor(Monitor):
    """
    A monitor for measuring the CPU usage of tritonserver during inference
    """

    cpu_metrics = {CPUAvailableRAM, CPUUsedRAM}

    def __init__(self, server, frequency, metrics):
        """
        Parameters
        ----------
        server : TritonServer
            A handle to the TritonServer
        frequency : float
            How often the metrics should be monitored.
        metrics : list
            A list of Record objects that will be monitored.
        """

        super().__init__(frequency, metrics)
        self._cpu_memory_records = []
        self._server = server

    def is_monitoring_connected(self) -> bool:
        return True

    def _monitoring_iteration(self):
        """
        Get memory info of process and 
        append
        """
        if (CPUUsedRAM in self._metrics) or (CPUAvailableRAM in self._metrics):
            used_mem, free_mem = self._server.cpu_stats()
            if CPUUsedRAM in self._metrics:
                self._cpu_memory_records.append(CPUUsedRAM(value=used_mem))
            if CPUAvailableRAM in self._metrics:
                self._cpu_memory_records.append(CPUAvailableRAM(value=free_mem))

    def _collect_records(self):
        """
        Returns
        -------
        List of Records 
            the records metrics
        """

        return self._cpu_memory_records
