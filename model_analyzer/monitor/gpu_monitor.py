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

import logging

from .monitor import Monitor
import numba.cuda

from model_analyzer.device.gpu_device_factory import GPUDeviceFactory
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

logger = logging.getLogger(__name__)


class GPUMonitor(Monitor):
    """
    Monitor abstract class is a parent class used for monitoring devices.
    """

    def __init__(self, gpus, frequency, metrics):
        """
        Parameters
        ----------
        gpus : list
            A list of strings containing GPU UUIDs.
        frequency : float
            How often the metrics should be monitored.
        metrics : list
            A list of Record objects that will be monitored.

        Raises
        ------
        TritonModelAnalyzerExcpetion
        """

        super().__init__(frequency, metrics)

        self._gpus = []

        if len(gpus) == 1 and gpus[0] == 'all':
            cuda_devices = numba.cuda.list_devices()
            if len(cuda_devices) == 0:
                raise TritonModelAnalyzerException(
                    "No GPUs are visible by CUDA. Make sure that 'nvidia-smi'"
                    " output shows available GPUs. If you are using Model"
                    " Analyzer inside a container, ensure that you are"
                    " launching the container with the"
                    " appropriate '--gpus' flag")
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
        logger.info(
            f'Using GPU(s) with UUID(s) = {{ {gpu_uuids_str} }} for profiling.')
