#!/usr/bin/env python3

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

import logging
from math import log2
from typing import Generator, List, Optional

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.constants import (
    LOGGER_NAME,
    THROUGHPUT_MINIMUM_CONSECUTIVE_INFERENCE_LOAD_TRIES,
    THROUGHPUT_MINIMUM_GAIN,
)
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

logger = logging.getLogger(LOGGER_NAME)


class InferenceLoadSearch:
    """
    Generates the next inference load value to use when searching through
    RunConfigMeasurements for the best value (according to the users objective)
      - Will sweep from by powers of two from min to max inference load
      - If the user specifies a constraint, the algorithm will perform a binary search
        around the boundary if the constraint is violated

    Invariant: It is necessary for the user to add new measurements as they are taken
    """

    def __init__(
        self,
        config: ConfigCommandProfile,
        model_parameters: dict = {},
        skip_inference_load_sweep: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        config: ConfigCommandProfile
            Profile configuration information
        skip_inference_load_sweep: bool
            If true, skips the inference load sweep and only does the binary search
        """
        self._skip_inference_load_sweep = skip_inference_load_sweep
        self._inference_load_is_request_rate = config.is_request_rate_specified(
            model_parameters
        )

        if self._inference_load_is_request_rate:
            self._min_inference_load_index = int(
                log2(config.run_config_search_min_request_rate)
            )
            self._max_inference_load_index = int(
                log2(config.run_config_search_max_request_rate)
            )

        else:
            self._min_inference_load_index = int(
                log2(config.run_config_search_min_concurrency)
            )
            self._max_inference_load_index = int(
                log2(config.run_config_search_max_concurrency)
            )

        self._max_binary_search_steps = config.run_config_search_max_binary_search_steps

        self._run_config_measurements: List[Optional[RunConfigMeasurement]] = []
        self._inference_loads: List[int] = []
        self._last_failing_inference_load = 0
        self._last_passing_inference_load = 0

    def add_run_config_measurement(
        self, run_config_measurement: Optional[RunConfigMeasurement]
    ) -> None:
        """
        Adds a new RunConfigMeasurement
        Invariant: Assumed that RCMs are added in the same order they are measured
        """
        self._run_config_measurements.append(run_config_measurement)

    def search_inference_loads(self) -> Generator[int, None, None]:
        """
        First performs an inference load sweep, and then, if necessary, perform
        a binary search around the point where the constraint was violated
        """
        yield from self._perform_inference_load_sweep()

        if self._was_constraint_violated():
            yield from self._perform_binary_search()

    def _perform_inference_load_sweep(self) -> Generator[int, None, None]:
        for inference_load in (
            2**i
            for i in range(
                self._min_inference_load_index, self._max_inference_load_index + 1
            )
        ):
            if self._should_continue_inference_load_sweep():
                self._inference_loads.append(inference_load)
                yield inference_load
            else:
                # We can't actually skip the sweep because the results need to be added
                # but, we can suppress the logging messages
                if not self._skip_inference_load_sweep:
                    if self._inference_load_is_request_rate:
                        logger.info(
                            "Terminating request rate sweep - throughput is decreasing"
                        )
                    else:
                        logger.info(
                            "Terminating concurrency sweep - throughput is decreasing"
                        )
                    return

    def _should_continue_inference_load_sweep(self) -> bool:
        self._check_measurement_count()

        if not self._are_minimum_tries_reached():
            return True
        else:
            return not self._has_objective_gain_saturated()

    def _check_measurement_count(self) -> None:
        if len(self._run_config_measurements) != len(self._inference_loads):
            raise TritonModelAnalyzerException(
                f"Internal Measurement count: {self._inference_loads}, doesn't match number "
                f"of measurements added: {len(self._run_config_measurements)}."
            )

    def _are_minimum_tries_reached(self) -> bool:
        if (
            len(self._run_config_measurements)
            < THROUGHPUT_MINIMUM_CONSECUTIVE_INFERENCE_LOAD_TRIES
        ):
            return False
        else:
            return True

    def _has_objective_gain_saturated(self) -> bool:
        gain = self._calculate_gain()
        return gain < THROUGHPUT_MINIMUM_GAIN

    def _calculate_gain(self) -> float:
        first_rcm = self._run_config_measurements[
            -THROUGHPUT_MINIMUM_CONSECUTIVE_INFERENCE_LOAD_TRIES
        ]

        best_rcm = self._get_best_rcm()

        # These cover the cases where we don't get a result from PA
        if not first_rcm and not best_rcm:
            return 0
        if not first_rcm:
            return 1
        elif not best_rcm:
            return -1
        else:
            gain = first_rcm.compare_measurements(best_rcm)

        return gain

    def _get_best_rcm(self) -> Optional[RunConfigMeasurement]:
        # Need to remove entries (None) with no result from PA before sorting
        pruned_rcms = [
            rcm
            for rcm in self._run_config_measurements[
                -THROUGHPUT_MINIMUM_CONSECUTIVE_INFERENCE_LOAD_TRIES:
            ]
            if rcm
        ]
        best_rcm = max(pruned_rcms) if pruned_rcms else None

        return best_rcm

    def _was_constraint_violated(self) -> bool:
        for i in range(len(self._run_config_measurements) - 1, 1, -1):
            if self._at_constraint_failure_boundary(i):
                self._last_failing_inference_load = self._inference_loads[i]
                self._last_passing_inference_load = self._inference_loads[i - 1]
                return True

        if (
            self._run_config_measurements[0]
            and not self._run_config_measurements[0].is_passing_constraints()
        ):
            self._last_failing_inference_load = self._inference_loads[i]
            self._last_passing_inference_load = 0
            return True
        else:
            return False

    def _at_constraint_failure_boundary(self, index: int) -> bool:
        if (
            not self._run_config_measurements[index]
            or not self._run_config_measurements[index - 1]
        ):
            return False

        at_failure_boundary = (
            not self._run_config_measurements[  # type: ignore
                index
            ].is_passing_constraints()
            and self._run_config_measurements[
                index - 1  # type: ignore
            ].is_passing_constraints()
        )

        return at_failure_boundary

    def _perform_binary_search(self) -> Generator[int, None, None]:
        # This is needed because we are going to restart the search from the
        # inference_load that failed - so we expect this to be at the end of the list
        self._inference_loads.append(self._last_failing_inference_load)

        for i in range(0, self._max_binary_search_steps):
            inference_load = self._determine_next_binary_inference_load()

            if inference_load != self._inference_loads[-1]:
                self._inference_loads.append(inference_load)
                yield inference_load

    def _determine_next_binary_inference_load(self) -> int:
        if not self._run_config_measurements[-1]:
            return 0

        if self._run_config_measurements[-1].is_passing_constraints():
            self._last_passing_inference_load = self._inference_loads[-1]
            inference_load = int(
                (self._last_failing_inference_load + self._inference_loads[-1]) / 2
            )
        else:
            self._last_failing_inference_load = self._inference_loads[-1]
            inference_load = int(
                (self._last_passing_inference_load + self._inference_loads[-1]) / 2
            )

        return inference_load
