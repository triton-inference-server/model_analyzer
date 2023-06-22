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

from typing import List

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.result.run_config_result_comparator import RunConfigResultComparator
from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

from bisect import insort
from functools import total_ordering
import logging

logger = logging.getLogger(LOGGER_NAME)


@total_ordering
class RunConfigResult:
    """
    A class that represents the group of measurements (result) from
    a single RunConfig. This RunConfigResult belongs
    to a particular ResultTable
    """

    def __init__(self, model_name: str, run_config: RunConfig,
                 comparator: RunConfigResultComparator,
                 constraint_manager: ConstraintManager):
        """
        Parameters
        ----------
        model_name: str
            The name of the model
        run_config : RunConfig
        comparator : RunConfigResultComparator
            An object whose compare function receives two
            RunConfigResults and returns 1 if the first is better than
            the second, 0 if they are equal and -1
            otherwise
        constraint_manager: ConstraintManager
            The object that handles processing and applying
            constraints on a given measurements
        """

        self._model_name = model_name
        self._run_config = run_config
        self._comparator = comparator
        self._constraint_manager = constraint_manager

        # Heaps
        self._measurements: List[RunConfigMeasurement] = []
        self._passing_measurements: List[RunConfigMeasurement] = []
        self._failing_measurements: List[RunConfigMeasurement] = []

    def model_name(self):
        """
        Returns
        -------
        str
            Returns the name of the model corresponding to this RunConfigResult
        """

        return self._model_name

    def run_config(self):
        """ 
        Returns
        -------
        RunConfig
            associated with this result 
        """
        return self._run_config

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

        insort(self._measurements, run_config_measurement)

        if self._constraint_manager.satisfies_constraints(
                run_config_measurement):
            insort(self._passing_measurements, run_config_measurement)
        else:
            insort(self._failing_measurements, run_config_measurement)

    def run_config_measurements(self):
        """
        Returns
        -------
        list
            of RunConfigMeasurements in this RunConfigResult
        """
        return [measurement for measurement in reversed(self._measurements)]

    def passing_measurements(self):
        """
        Returns
        -------
        list
            of passing measurements in this RunConfigResult
        """

        return [
            passing_measurement
            for passing_measurement in reversed(self._passing_measurements)
        ]

    def failing_measurements(self):
        """
        Returns
        -------
        list
            of failing measurements in this RunConfigResult
        """

        return [
            failing_measurement
            for failing_measurement in reversed(self._failing_measurements)
        ]

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

            return [
                failing_measurement for failing_measurement in reversed(
                    self._failing_measurements[-n:])
            ]

        if n > len(self._passing_measurements):
            logger.warning(
                f"Requested top {n} RunConfigMeasurements, but "
                f"found only {len(self._passing_measurements)}. "
                "Showing all available measurements for this config.")

        return [
            passing_measurement
            for passing_measurement in reversed(self._passing_measurements[-n:])
        ]

    def __lt__(self, other):
        """
        Checks whether this RunConfigResult is better
        than other

        If True, this means this RunConfigResult is better
        than the other.
        """

        return self._comparator.is_better_than(self, other)
