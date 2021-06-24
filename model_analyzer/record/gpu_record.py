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

from .record import Record


class GPURecord(Record):
    """
    This is a base class for any
    GPU based record
    """
    def __init__(self, value, device=None, timestamp=0):
        """
        Parameters
        ----------
        value : float
            The value of the GPU metrtic
        device : GPUDevice
            The  GPU device this metric is associated
            with.
        timestamp : int
            The timestamp for the record in nanoseconds
        """

        super().__init__(value, timestamp)
        self._device = device

    def device(self):
        """
        Returns
        -------
        GPUDevice
            handle for this metric
        """

        return self._device

    @classmethod
    def from_dict(cls, record_dict):
        record = cls(0)
        for key in ['_value', '_timestamp', '_device']:
            if key in record_dict:
                setattr(record, key, record_dict[key])
        return record
