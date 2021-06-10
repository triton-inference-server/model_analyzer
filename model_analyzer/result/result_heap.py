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
import logging


class ResultHeap:
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
        True if this heap has no results
        False otherwise
        """

        return not bool(self._sorted_results)

    def results(self):
        """
        Returns
        -------
        All the results in this result heap
        """

        return self._sorted_results

    def add_result(self, result):
        """
        Adds a result to the result 
        heaps

        Parameters
        ----------
        result: ModelResult
            The result to be added
        """

        heapq.heappush(self._sorted_results, result)
        if result.failing():
            heapq.heappush(self._failing_results, result)
        else:
            heapq.heappush(self._passing_results, result)

    def next_best_result(self):
        """
        Removes and returns the 
        next best item in this 
        result heap

        Returns
        -------
        ModelResult
            The next best result in this heap
        """

        return heapq.heappop(self._sorted_results)

    def top_n_results(self, n):
        """
        Parameters
        ----------
        n : int
            The number of  top results
            to retrieve, get all if n==-1

        Returns
        -------
        list of ModelResults
            The n best results for this model,
            must all be passing results
        """

        if len(self._passing_results) == 0:
            logging.warning(
                f"Requested top {n} configs, but none satisfied constraints. "
                "Showing available constraint failing configs for this model.")

            if n == -1:
                return heapq.nsmallest(len(self._failing_results),
                                       self._failing_results)
            if n > len(self._failing_results):
                logging.warning(
                    f"Requested top {n} failing configs, "
                    f"but found only {len(self._failing_results)}. "
                    "Showing all available constraint failing configs for this model."
                )
            return heapq.nsmallest(min(n, len(self._failing_results)),
                                   self._failing_results)

        if n == -1:
            return heapq.nsmallest(len(self._passing_results),
                                   self._passing_results)
        if n > len(self._passing_results):
            logging.warning(
                f"Requested top {n} configs, "
                f"but found only {len(self._passing_results)} passing configs. "
                "Showing all available constraint satisfying configs for this model."
            )

        return heapq.nsmallest(min(n, len(self._passing_results)),
                               self._passing_results)
