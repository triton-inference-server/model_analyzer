#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from pynvml import *
from multiprocessing.pool import ThreadPool
import time

from model_analyzer.monitor.record.memory import MemoryRecord
from model_analyzer.monitor.record.collector import RecordCollector
from model_analyzer.monitor import Monitor


class NVMLMonitor(Monitor):
    """Use NVML to monitor GPU metrics
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

        # FIXME Handle the case when the number of GPUs is larger than one and
        # CUDA_VISIBLE_DEVICES is set
        self._nvml_handles = []
        for i, gpu in enumerate(self._gpus):
            self._nvml_handles.append(nvmlDeviceGetHandleByIndex(i))

        # Is the background thread active
        self._thread_active = False

        # Background thread collecting results
        self._thread = None

        # Thread pool
        self._thread_pool = ThreadPool(processes=1)

    def _monitoring_loop(self, tags):
        self._thread_active = True
        frequency = self._frequency

        record_collector = RecordCollector()
        while self._thread_active:
            begin = time.time()
            for tag in tags:
                if tag == 'memory':
                    for handle in self._nvml_handles:
                        memory = nvmlDeviceGetMemoryInfo(handle)
                        memory_record = MemoryRecord(memory.used, memory.total,
                                                     memory.free)
                        record_collector.insert(memory_record)

            duration = time.time() - begin
            if duration < frequency:
                time.sleep(frequency - duration)

        return record_collector

    def start_recording_metrics(self, tags):
        """Start recording a list of tags

        Parameters
        ----------
        tags : list
            Name of the metrics to be collected. Options include
            *memory* and *compute*.
        """
        self._thread = self._thread_pool.apply_async(self._monitoring_loop,
                                                     (tags, ))

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
        """Destroy the NVMLMonitor. This function must be called
        in order to appropriately deallocate the resources.
        """
        self._thread_pool.terminate()
        self._thread_pool.close()
