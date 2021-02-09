# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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


@total_ordering
class Measurement:
    """
    Encapsulates the set of metrics obtained from a single 
    perf_analyzer run
    """

    def __init__(self, gpu_data, non_gpu_data, perf_config, comparator):
        """
        gpu_data : dict of list of RecordTypes
            These are the values from the monitors that have a GPU ID
            associated with them
        non_gpu_data : list of RecordTypes
            These do not have a GPU ID associated with them
        perf_config : PerfAnalyzerConfig
            The perf config that was used for the perf run that generated
            this data data
        comparator : ResultComparator
            Handle for ResultComparator that knows how to order measurements
        """

        self._gpu_data = gpu_data
        self._non_gpu_data = non_gpu_data
        self._perf_config = perf_config
        self._comparator = comparator

    def gpu_data(self):
        """
        Returns the GPU ID specific measurement
        """

        return self._gpu_data

    def non_gpu_data(self):
        """
        Returns the non GPU ID specific measurement
        """

        return self._non_gpu_data

    def perf_config(self):
        """
        Return the PerfAnalyzerConfig
        used to get this measurement
        """

        return self._perf_config

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
