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

import time
from .gpu_record import GPURecord


class GPUUsedMemory(GPURecord):
    """
    The used memory in the GPU.
    """

    def __init__(self, device, used_mem, timestamp):
        """
        Parameters
        ----------
        device : GPUDevice
            The GPU device this metric is associated
            with.
        used_mem : float
            GPU used memory
        timestamp : int
            The timestamp for the record in nanoseconds
        """

        super().__init__(device, used_mem, timestamp)

    @staticmethod
    def header():
        """
        Returns
        -------
        str
            The full name of the
            metric.
        """

        return "GPU Used Memory(MB)"
