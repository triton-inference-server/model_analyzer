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

from model_analyzer.constants import LOGGER_NAME
from .constraint_manager import ConstraintManager

import heapq
from functools import total_ordering
import logging

logger = logging.getLogger(LOGGER_NAME)


@total_ordering
class RunConfigResult:
    """
    A class that represents the group of measurements from
    a single RunConfig. This RunConfigResult belongs
    to a particular ResultTable
    """

    def __init__(self, model_name, model_configs, comparator, constraints=None):
        """
        Parameters
        ----------
        model_name: str
            The name of the model
        model_configs : list of ModelConfigs
            The list of model configs corresponding to this RunConfig
        comparator : RunConfigResultComparator
            An object whose compare function receives two
            RunConfigResults and returns 1 if the first is better than
            the second, 0 if they are equal and -1
            otherwise
        constraints: dict
            Used to determine if a RunConfigResult passes or fails
            metric: (constraint_type:constraint_value)
        """

        self._model_name = model_name
        self._model_configs = model_configs
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
            Returns the name of the model corresponding to this RunConfigResult
        """

        return self._model_name

    def model_configs(self):
        """
        Returns
        -------
        List of ModelConfigs
            Returns the list model_configs associated with this RunConfigResult
        """

        return self._model_configs

    def failing(self):
        """
        Returns
        -------
        bool
            Returns true if there are no passing RunConfigMeasurements
        """

        if not self._passing_measurements:
            return True
        return False

    def add_run_config_measurement(self, run_config_measurement):
        """
        This function checks whether a RunConfigMeasurement
        passes the constraints and adds the measurements to 
        the corresponding heap

        Parameters
        ----------
        run_config_measurement : RunConfigMeasurement
            The profiled RunConfigMeasurement
        """

        heapq.heappush(self._measurements, run_config_measurement)
        if ConstraintManager.check_constraints(self._constraints,
                                               run_config_measurement):
            heapq.heappush(self._passing_measurements, run_config_measurement)
        else:
            heapq.heappush(self._failing_measurements, run_config_measurement)

    def run_config_measurements(self):
        """
        Returns
        -------
        list
            of RunConfigMeasurements in this RunConfigResult
        """

        return self._measurements

    def passing_measurements(self):
        """
        Returns
        -------
        list
            of passing measurements in this RunConfigResult
        """

        return self._passing_measurements

    def failing_measurements(self):
        """
        Returns
        -------
        list
            of failing measurements in this RunConfigResult
        """

        return self._failing_measurements

    def top_n_measurements(self, n):
        """
        Parameters
        ----------
        n : int
            The number of top RunConfigMeasurements 
            to retrieve

        Returns
        -------
        list of RunConfigMeasurements
            The top n RunConfigMeasurements
        """

        if len(self._passing_measurements) == 0:
            logger.warning(
                f"Requested top {n} RunConfigMeasurements, but none satisfied constraints. "
                "Showing available constraint failing measurements for this config."
            )

            if n > len(self._failing_measurements):
                logger.warning(
                    f"Requested top {n} failing RunConfigMeasurements, "
                    f"but found only {len(self._failing_measurements)}. "
                    "Showing all available constraint failing measurements for this config."
                )
            return heapq.nsmallest(min(n, len(self._failing_measurements)),
                                   self._failing_measurements)

        if n > len(self._passing_measurements):
            logger.warning(
                f"Requested top {n} RunConfigMeasurements, but "
                f"found only {len(self._passing_measurements)}. "
                "Showing all available measurements for this config.")

        return heapq.nsmallest(min(n, len(self._passing_measurements)),
                               self._passing_measurements)

    def __lt__(self, other):
        """
        Checks whether this RunConfigResult is better
        than other

        If True, this means this RunConfigResult is better
        than the other.
        """

        return self._comparator.is_better_than(self, other)
