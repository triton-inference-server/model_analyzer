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

from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
import numba.cuda
import time
import logging

from model_analyzer.device.gpu_device_factory import GPUDeviceFactory
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

logger = logging.getLogger(__name__)


class Monitor(ABC):
    """
    Monitor abstract class is a parent class used for monitoring devices.
    """
    def __init__(self, gpus, frequency, tags):
        """
        Parameters
        ----------
        gpus : list
            A list of strings containing GPU UUIDs.
        frequency : float
            How often the metrics should be monitored.
        tags : list
            A list of Record objects that will be monitored.

        Raises
        ------
        TritonModelAnalyzerExcpetion
            If the GPU cannot be found, the exception will be raised.
        """

        self._frequency = frequency
        self._gpus = []

        if len(gpus) == 1 and gpus[0] == 'all':
            cuda_devices = numba.cuda.list_devices()
            if len(cuda_devices) == 0:
                raise TritonModelAnalyzerException(
                    "No GPUs are visible by CUDA. Make sure that 'nvidia-smi'"
                    " output shows available GPUs. If you are using Model"
                    " Analyzer inside a container, ensure that you are"
                    " launching the container with the"
                    " appropriate '--gpus' flag"
                )
            for gpu in cuda_devices:
                gpu_device = GPUDeviceFactory.create_device_by_cuda_index(
                    gpu.id)
                self._gpus.append(gpu_device)
        else:
            for gpu in gpus:
                gpu_device = GPUDeviceFactory.create_device_by_uuid(gpu)
                self._gpus.append(gpu_device)

        gpu_uuids = []
        for gpu in self._gpus:
            gpu_uuids.append(str(gpu.device_uuid(), encoding='ascii'))
        gpu_uuids_str = ','.join(gpu_uuids)
        logger.info(f'Using GPU(s) with UUID(s) = {{ {gpu_uuids_str} }} for the analysis.')

        # Is the background thread active
        self._thread_active = False

        # Background thread collecting results
        self._thread = None

        # Thread pool
        self._thread_pool = ThreadPool(processes=1)
        self._tags = tags

    def _monitoring_loop(self):
        frequency = self._frequency

        while self._thread_active:
            begin = time.time()
            # Monitoring iteration implemented by each of the subclasses
            self._monitoring_iteration()

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

        self._thread_active = True
        self._thread = self._thread_pool.apply_async(self._monitoring_loop)

    def stop_recording_metrics(self):
        """
        Stop recording metrics. This will stop monitring all the metrics.
        """

        if not self._thread_active:
            raise TritonModelAnalyzerException(
                'start_recording_metrics should be called before\
                     stop_recording_metrics')

        self._thread_active = False
        self._thread = None

        return self._collect_records()
