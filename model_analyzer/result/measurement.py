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

from functools import total_ordering
import logging
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig

from model_analyzer.record.record import RecordType


@total_ordering
class Measurement:
    """
    Encapsulates the set of metrics obtained from a single
    perf_analyzer run
    """

    def __init__(self, gpu_data, non_gpu_data, perf_config):
        """
        gpu_data : dict of list of Records
            These are the values from the monitors that have a GPU UUID
            associated with them
        non_gpu_data : list of Records
            These do not have a GPU UUID associated with them
        perf_config : PerfAnalyzerConfig
            The perf config that was used for the perf run that generated
            this data data
        """

        # average values over all GPUs
        self._gpu_data = gpu_data
        self._non_gpu_data = non_gpu_data
        self._perf_config = perf_config

        self._avg_gpu_data = self._average_list(list(self._gpu_data.values()))
        self._gpu_data_from_tag = {
            type(metric).tag: metric for metric in self._avg_gpu_data
        }
        self._non_gpu_data_from_tag = {
            type(metric).tag: metric for metric in self._non_gpu_data
        }

    @classmethod
    def from_dict(cls, measurement_dict):
        measurement = Measurement({}, [], None)

        # Deserialize gpu_data
        for gpu_uuid, gpu_data_list in measurement_dict['_gpu_data'].items():
            metric_list = []
            for [tag, record_dict] in gpu_data_list:
                record_type = RecordType.get(tag)
                record = record_type.from_dict(record_dict)
                metric_list.append(record)
            measurement._gpu_data[gpu_uuid] = metric_list

        # non gpu data
        measurement._non_gpu_data = []
        for [tag, record_dict] in measurement_dict['_non_gpu_data']:
            record_type = RecordType.get(tag)
            record = record_type.from_dict(record_dict)
            measurement._non_gpu_data.append(record)

        # perf config
        measurement._perf_config = PerfAnalyzerConfig.from_dict(
            measurement_dict['_perf_config'])

        # Compute contigent data structures
        measurement._avg_gpu_data = measurement._average_list(
            list(measurement._gpu_data.values()))
        measurement._gpu_data_from_tag = {
            type(metric).tag: metric for metric in measurement._avg_gpu_data
        }
        measurement._non_gpu_data_from_tag = {
            type(metric).tag: metric for metric in measurement._non_gpu_data
        }

        return measurement

    def set_result_comparator(self, comparator):
        """
        Sets result comparator for this
        measurement
        Parameters
        ----------
        comparator : ResultComparator
            Handle for ResultComparator that knows how to order measurements
        """

        self._comparator = comparator

    def data(self):
        """
        Returns
        -------
        list of records
            the metric values in this measurement
        """

        return self._avg_gpu_data + self._non_gpu_data

    def gpu_data(self):
        """
        Returns the GPU specific measurement
        """

        return self._gpu_data

    def non_gpu_data(self):
        """
        Returns the non GPU specific measurement
        """

        return self._non_gpu_data

    def perf_config(self):
        """
        Return the PerfAnalyzerConfig
        used to get this measurement
        """

        return self._perf_config

    def get_metric(self, tag):
        """
        Parameters
        ----------
        tag : str
            A human readable tag that corresponds
            to a particular metric

        Returns
        -------
        Record
            metric Record corresponding to
            the tag, in this measurement, None
            if tag not found.
        """

        if tag in self._non_gpu_data_from_tag:
            return self._non_gpu_data_from_tag[tag]
        elif tag in self._gpu_data_from_tag:
            return self._gpu_data_from_tag[tag]
        else:
            logging.warning(f"No metric corresponding to tag '{tag}' "
                            "found in measurement. Possibly comparing "
                            "measurements across devices.")
            return None

    def get_metric_value(self, tag, default_value=0):
        """
        Parameters
        ----------
        tag : str
            A human readable tag that corresponds
            to a particular metric
        default_value : any
            Value to return if tag is not found

        Returns
        -------
        Record
            Value of the metric Record corresponding 
            to the tag, in this measurement, 
            default_value if tag not found.
        """
        metric = self.get_metric(tag)
        if metric is None:
            return default_value
        return metric.value()

    def get_parameter(self, tag):
        """
        Parameters
        ----------
        tag : str
            A human readable tag that corresponds
            to a particular parameter

        Returns
        -------
        value
            metric Record value corresponding to
            the tag, in this measurement, None
            if tag not found
        """

        if tag.replace('_', '-') in self.perf_config():
            return self.perf_config()[tag.replace('_', '-')]
        else:
            logging.warning(f"No parameter corresponding to tag '{tag}' "
                            "found in measurement. Possibly comparing "
                            "measurements across devices.")
            return None

    def gpus_used(self):
        """
        Returns
        -------
        list of ints
            list of device IDs used in this measurement
        """

        return list(self._gpu_data.keys())

    def __eq__(self, other):
        """
        Check whether two sets of measurements are equivalent
        """

        return self._comparator.compare_measurements(self, other) == 0

    def __lt__(self, other):
        """
        Checks whether a measurement is better than
        another

        If True, this means this measurement is better
        than the other.
        """

        # seems like this should be == -1 but we're using a min heap
        return self._comparator.compare_measurements(self, other) == 1

    def _average_list(self, row_list):
        """
        Average a 2d list
        """

        if not row_list:
            return row_list
        else:
            N = len(row_list)
            d = len(row_list[0])
            avg = [0 for _ in range(d)]
            for i in range(d):
                avg[i] = (sum([row_list[j][i] for j in range(1, N)],
                              start=row_list[0][i]) * 1.0) / N
            return avg
