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

from typing import List, Optional

from copy import deepcopy
from bisect import insort

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.result.run_config_result import RunConfigResult

import logging

logger = logging.getLogger(LOGGER_NAME)


class SortedResults:
    """
    A data structure used by the result manager 
    to store and sort results
    """

    GET_ALL_RESULTS = -1

    def __init__(self):
        self._run_config_results = []

    def results(self) -> List[RunConfigResult]:
        """
        Returns
        -------
        All the results
        """

        self._run_config_results.sort()
        return self._run_config_results

    def add_result(self, run_config_result: RunConfigResult):
        """
        Adds a run_config_result to the result lists
        This can either be a new result or new measurements added
        to an existing result

        Parameters
        ----------
        result: ModelResult
            The result to be added
        """

        existing_run_config_result = self._find_existing_run_config_result(
            run_config_result)

        if existing_run_config_result:
            self._add_measurements_to_existing_run_config_result(
                existing_run_config_result, run_config_result)
        else:
            self._add_new_run_config_result(run_config_result)

    def top_n_results(self, n: int) -> List[RunConfigResult]:
        """
        Parameters
        ----------
        n : int
            The number of  top results
            to retrieve

        Returns
        -------
        list of RunConfigResults
            The n best results for this model,
            must all be passing results
        """
        self._run_config_results.sort()
        passing_results, failing_results = self._create_passing_and_failing_lists(
        )

        if len(passing_results) == 0:
            return self._get_top_n_failing_results(failing_results, n)
        else:
            return self._get_top_n_passing_results(passing_results, n)

    def _find_existing_run_config_result(
            self,
            run_config_result: RunConfigResult) -> Optional[RunConfigResult]:
        if not run_config_result.run_config():
            return None

        for rcr in self._run_config_results:
            if run_config_result.run_config().model_variants_name(
            ) == rcr.run_config().model_variants_name():
                return rcr

        return None

    def _add_measurements_to_existing_run_config_result(
            self, existing_run_config_result: RunConfigResult,
            new_run_config_result: RunConfigResult):
        for rcm in new_run_config_result.run_config_measurements():
            existing_run_config_result.add_run_config_measurement(rcm)

    def _add_new_run_config_result(self, run_config_result: RunConfigResult):
        new_run_config_result = deepcopy(run_config_result)

        self._run_config_results.append(new_run_config_result)

    def _create_passing_and_failing_lists(self):
        passing = []
        failing = []
        for rcr in self._run_config_results:
            if rcr.failing():
                failing.append(rcr)
            else:
                passing.append(rcr)

        return passing, failing

    def _get_top_n_failing_results(self, failing_results: List[RunConfigResult],
                                   n: int) -> List[RunConfigResult]:
        logger.warning(
            f"Requested top {n} configs, but none satisfied constraints. "
            "Showing available constraint failing configs for this model.")

        if n == SortedResults.GET_ALL_RESULTS:
            return failing_results
        if n > len(failing_results):
            logger.warning(
                f"Requested top {n} failing configs, "
                f"but found only {len(failing_results)}. "
                "Showing all available constraint failing configs for this model."
            )

        result_len = min(n, len(failing_results))
        return failing_results[0:result_len]

    def _get_top_n_passing_results(self, passing_results: List[RunConfigResult],
                                   n: int) -> List[RunConfigResult]:
        if n == SortedResults.GET_ALL_RESULTS:
            return passing_results
        if n > len(passing_results):
            logger.warning(
                f"Requested top {n} configs, "
                f"but found only {len(passing_results)} passing configs. "
                "Showing all available constraint satisfying configs for this model."
            )

        result_len = min(n, len(passing_results))
        return passing_results[0:result_len]
