# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import heapq
from functools import total_ordering


@total_ordering
class RunResult:
    """
    A class that represents the result of
    a single run. This RunResult belongs
    to a particular ResultTable
    """

    def __init__(self, run_config, comparator):
        """
        Parameters
        ----------
        run_config : RunConfig
            The model config corresponding with the current
            RunResult
        comparator : ResultComparator
            An object whose compare function receives two
            results and returns 1 if the first is better than
            the second, 0 if they are equal and -1
            otherwise
        """

        self._run_config = run_config
        self._comparator = comparator

        # Heap
        self._measurements = []

    def add_measurement(self, measurement):
        """
        This function adds model inference
        measurements to the result

        Parameters
        ----------
        measurement : Measurement
            The measurement from the metrics manager,
            actual values from the monitors
        """

        heapq.heappush(self._measurements, measurement)

    def measurements(self):
        """
        Returns
        -------
        (list, list)
            gpu_specific, and non gpu specific measurements
            respectively.
        """

        return self._measurements

    def top_n_measurements(self, n):
        """
        Parameters
        ----------
        n : int
            The number of  top results 
            to retrieve

        Returns
        -------
        The top n measurements in
        this result
        """

        return heapq.nsmallest(n, self._measurements)

    def run_config(self):
        """
        Returns
        -------
        RunConfig
            returns the run_config associated with this
            RunResult
        """

        return self._run_config

    def __eq__(self, other):
        """
        Checks for the equality of this and
        another RunResult
        """

        return (self._comparator.compare_results(self, other) == 0)

    def __lt__(self, other):
        """
        Checks whether this RunResult is better
        than other

        If True, this means this result is better
        than the other.
        """

        # Seems like this should be == -1 but we are using a min heap
        return (self._comparator.compare_results(self, other) == 1)
