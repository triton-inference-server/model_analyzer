# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, Tuple, Optional, Generator

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

from math import log2

import logging
from model_analyzer.constants import LOGGER_NAME, THROUGHPUT_MINIMUM_GAIN, THROUGHPUT_MINIMUM_CONSECUTIVE_CONCURRENCY_TRIES

logger = logging.getLogger(LOGGER_NAME)


class ConcurrencySearch():
    """
    Determines the next concurrency value to use when searching through
    a RunConfigMeasurement for the best value (according to the users objective)
      - Will sweep from by powers of two from min to max concurrency
      - If the user specifies a constraint, the algorithm will perform a binary search 
        around the boundary if/when the constraint is violated
    """

    def __init__(self, config: ConfigCommandProfile) -> None:
        """
        Parameters
        ----------
        config: ConfigCommandProfile
            Profile configuration information
        """
        self._config = config

        self._run_config_measurements = []
        self._binary_search_required = False

    def add_run_config_measurement(
            self, run_config_measurement: RunConfigMeasurement) -> None:
        self._run_config_measurements.append(run_config_measurement)

    def search_concurrencies(self) -> Generator[int, None, None]:
        yield from self._perform_exponential_concurrency_sweep()

        if self._binary_search_required:
            yield from self._perform_binary_concurrency_search()

    def _perform_exponential_concurrency_sweep(
            self) -> Generator[int, None, None]:
        min_concurrency_index = int(
            log2(self._config.run_config_search_min_concurrency))
        max_concurrency_index = int(
            log2(self._config.run_config_search_max_concurrency))

        for concurrency in (
                2**i
                for i in range(min_concurrency_index, max_concurrency_index +
                               1)):
            if not self._has_objective_gain_saturated_or_constraint_violated():
                yield concurrency

    def _perform_binary_concurrency_search(self) -> Generator[int, None, None]:
        yield 0

    def _has_objective_gain_saturated_or_constraint_violated(self) -> bool:
        # FIXME: need to consider constraints first!
        if len(self._run_config_measurements
              ) < THROUGHPUT_MINIMUM_CONSECUTIVE_CONCURRENCY_TRIES:
            return False

        first_rcm = self._run_config_measurements[
            -THROUGHPUT_MINIMUM_CONSECUTIVE_CONCURRENCY_TRIES]
        best_rcm = max(self._run_config_measurements[
            -THROUGHPUT_MINIMUM_CONSECUTIVE_CONCURRENCY_TRIES:])

        gain = first_rcm.compare_measurements(best_rcm)

        return gain < THROUGHPUT_MINIMUM_GAIN
