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

from abc import ABC, ABCMeta, abstractmethod
import importlib


class RecordType(ABCMeta):
    """
    A metaclass that holds the instantiated Record types
    """

    record_types = {}

    def __new__(cls, name, base, namespace):
        """
        This function is called upon declaration
        of any classes of type RecordType
        """

        record_type = super().__new__(cls, name, base, namespace)

        # If record_type.tag is a string, register it here
        if isinstance(record_type.tag, str):
            cls.record_types[record_type.tag] = record_type
        return record_type

    @classmethod
    def get(cls, tag):
        """
        Parameters
        ----------
        tag : str
            tag that a record type has registered it classname with
        
        Returns
        -------
        The class of type RecordType correspoding to the tag
        """

        if tag not in cls.record_types:
            try:
                importlib.import_module('model_analyzer.record.%s' % tag)
            except ImportError as e:
                print(e)
        return cls.record_types[tag]


class Record(metaclass=RecordType):
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
    def header(aggregation_tag=None):
        """
        Parameters
        ----------
        aggregation_tag : str
            An optional tag that may be displayed 
            as part of the header indicating that 
            this record has been aggregated using 
            max, min or average etc. 
             
        Returns
        -------
        str
            The full name of the
            metric.
        """

    @property
    @abstractmethod
    def tag(self):
        """
        Returns
        -------
        str
            the name tag of the record type. 
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
