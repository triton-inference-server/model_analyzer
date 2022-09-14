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

    def __init__(self):
        self._sorted_results = []
        self._failing_results = []
        self._passing_results = []

    def empty(self):
        """
        Returns
        -------
        True if this list has no results
        False otherwise
        """

        return not bool(self._sorted_results)

    def results(self) -> List[RunConfigResult]:
        """
        Returns
        -------
        All the results in this result list
        """

        return self._sorted_results

    def add_result(self, result: RunConfigResult):
        """
        Adds a result to the result lists
        This can either be a new result or new measurements added
        to an existing result

        Parameters
        ----------
        result: ModelResult
            The result to be added
        """

        existing_run_config = self._find_existing_run_config(result)

        if existing_run_config:
            self._add_result_to_existing_run_config(existing_run_config, result)
        else:
            self._add_new_results(result)

    def next_best_result(self) -> RunConfigResult:
        """
        Removes and returns the 
        next best item in this 
        result list

        Returns
        -------
        ModelResult
            The next best result in this list
        """

        return self._sorted_results.pop(0)

    def top_n_results(self, n: int) -> List[RunConfigResult]:
        """
        Parameters
        ----------
        n : int
            The number of  top results
            to retrieve, get all if n==-1

        Returns
        -------
        list of RunConfigResults
            The n best results for this model,
            must all be passing results
        """
        if len(self._passing_results) == 0:
            logger.warning(
                f"Requested top {n} configs, but none satisfied constraints. "
                "Showing available constraint failing configs for this model.")

            if n == -1:
                return self._failing_results
            if n > len(self._failing_results):
                logger.warning(
                    f"Requested top {n} failing configs, "
                    f"but found only {len(self._failing_results)}. "
                    "Showing all available constraint failing configs for this model."
                )

            result_len = min(n, len(self._failing_results))
            return self._failing_results[0:result_len]

        if n == -1:
            return self._passing_results
        if n > len(self._passing_results):
            logger.warning(
                f"Requested top {n} configs, "
                f"but found only {len(self._passing_results)} passing configs. "
                "Showing all available constraint satisfying configs for this model."
            )

        result_len = min(n, len(self._passing_results))
        return self._passing_results[0:result_len]

    def _find_existing_run_config(
            self, result: RunConfigResult) -> Optional[RunConfigResult]:
        if not result.run_config():
            return None

        for rcr in self._sorted_results:
            if result.run_config().model_variants_name() == rcr.run_config(
            ).model_variants_name():
                return rcr

        return None

    def _add_result_to_existing_run_config(self,
                                           existing_run_config: RunConfigResult,
                                           result: RunConfigResult):
        for rcm in result.run_config_measurements():
            existing_run_config.add_run_config_measurement(rcm)

        self._sorted_results.sort()
        self._passing_results.sort()
        self._failing_results.sort()

    def _add_new_results(self, result: RunConfigResult):
        new_result = deepcopy(result)

        insort(self._sorted_results, new_result)
        if result.failing():
            insort(self._failing_results, new_result)
        else:
            insort(self._passing_results, new_result)
