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

from .constraint_manager import ConstraintManager

import heapq
import logging
from functools import total_ordering


@total_ordering
class ModelResult:
    """
    A class that represents the result of
    a single run. This ModelResult belongs
    to a particular ResultTable
    """

    def __init__(self, model_name, model_config, comparator, constraints=None):
        """
        Parameters
        ----------
        model_name: str
            The name of the model
        model_config : ModelConfig
            The model config corresponding to this run
        comparator : ResultComparator
            An object whose compare function receives two
            results and returns 1 if the first is better than
            the second, 0 if they are equal and -1
            otherwise
        constraints: dict
            keys are metrics and values are 
            constraint_type:constraint_value pairs
        """

        self._model_name = model_name
        self._model_config = model_config
        self._comparator = comparator
        self._constraints = constraints

        # Heaps
        self._measurements = []
        self._passing_measurements = []
        self._failing_measurements = []

    def model_name(self):
        """
        Returns
        -------
        str
            returns the name of the model whose ModelResult this is
        """

        return self._model_name

    def model_config(self):
        """
        Returns
        -------
        ModelConfig
            returns the model_config associated with this
            ModelResult
        """

        return self._model_config

    def failing(self):
        """
        Returns
        -------
        True if this is a failing result
        False if at least one measurement
        passes
        """

        if not self._passing_measurements:
            return True
        return False

    def add_measurement(self, measurement):
        """
        This function checks whether a measurement
        passes the constraints and adds model inference
        measurements to the corresponding heap

        Parameters
        ----------
        measurement : Measurement
            The measurement from the metrics manager,
            actual values from the monitors
        """

        heapq.heappush(self._measurements, measurement)
        if ConstraintManager.check_constraints(self._constraints, measurement):
            heapq.heappush(self._passing_measurements, measurement)
        else:
            heapq.heappush(self._failing_measurements, measurement)

    def measurements(self):
        """
        Returns
        -------
        list
            of measurements in this ModelResult
        """

        return self._measurements

    def passing_measurements(self):
        """
        Returns
        -------
        list
            of passing measurements in this ModelResult
        """

        return self._passing_measurements

    def failing_measurements(self):
        """
        Returns
        -------
        list
            of failing measurements in this ModelResult
        """

        return self._failing_measurements

    def top_n_measurements(self, n):
        """
        Parameters
        ----------
        n : int
            The number of  top measurements 
            to retrieve

        Returns
        -------
        list os Measurements
            The top n measurements in
            this result
        """

        if len(self._passing_measurements) == 0:
            logging.warn(
                f"Requested top {n} measurements, but none satisfied constraints. "
                "Showing available constraint failing measurements for this config."
            )

            if n > len(self._failing_measurements):
                logging.warn(
                    f"Requested top {n} failing measurements, "
                    f"but found only {len(self._failing_measurements)}. "
                    "Showing all available constraint failing measurements for this config."
                )
            return heapq.nsmallest(min(n, len(self._failing_measurements)),
                                   self._failing_measurements)

        if n > len(self._passing_measurements):
            logging.warn(f"Requested top {n} measurements, but "
                         f"found only {len(self._passing_measurements)}. "
                         "Showing all available measurements for this config.")

        return heapq.nsmallest(min(n, len(self._passing_measurements)),
                               self._passing_measurements)

    def __eq__(self, other):
        """
        Checks for the equality of this and
        another ModelResult
        """

        return (self._comparator.compare_results(self, other) == 0)

    def __lt__(self, other):
        """
        Checks whether this ModelResult is better
        than other

        If True, this means this result is better
        than the other.
        """

        # Seems like this should be == -1 but we are using a min heap
        return (self._comparator.compare_results(self, other) == 1)
