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


class Record(ABC):
    """
    This class is used for representing
    records
    """

    def __init__(self, value, timestamp):
        """
        Parameters
        ----------
        value : float or int
            The value of the GPU metrtic
        timestamp : int
            The timestamp for the record in nanoseconds
        """
        assert type(value) is float or type(value) is int
        assert type(timestamp) is int

        self._value = value
        self._timestamp = timestamp

    @staticmethod
    @abstractmethod
    def header():
        """
        Returns
        -------
        str
            The full name of the
            metric.
        """

    def value(self):
        """
        This method returns
        the value of recorded
        metric

        Returns
        -------
        float
            value of the metric
        """

        return self._value

    def timestamp(self):
        """
        This method should
        return the time
        at which the record
        was created.

        Returns
        -------
        float
            timestamp passed in during
            record creation
        """

        return self._timestamp
