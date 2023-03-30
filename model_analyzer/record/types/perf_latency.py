# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging

from model_analyzer.record.record import DecreasingRecord


@total_ordering
class PerfLatency(DecreasingRecord):
    """
    A record for perf_analyzer latency metric
    """

    tag = "perf_latency"
    _printed_deprecate_warning = False

    def __init__(self, value, timestamp=0):
        """
        Parameters
        ----------
        value : float
            the latency extracted from the perf analyzer output
        timestamp : float
            Elapsed time from start of program
        """

        super().__init__(value, timestamp)

    @classmethod
    def header(cls, aggregation_tag=False):
        """
        Parameters
        ----------
        aggregation_tag: bool
            An optional tag that may be displayed 
            as part of the header indicating that 
            this record has been aggregated using 
            max, min or average etc. 
             
        Returns
        -------
        str
            The full name of the
            metric.
        """

        cls._print_deprecate_warning()
        return "p99 Latency (ms)"

    def __eq__(self, other):
        """
        Allows checking for
        equality between two records
        """

        self._print_deprecate_warning()
        return self.value() == other.value()

    def __lt__(self, other):
        """
        Allows checking if 
        this record is less than 
        the other
        """

        self._print_deprecate_warning()
        return self.value() > other.value()

    def __add__(self, other):
        """
        Allows adding two records together
        to produce a brand new record.
        """

        self._print_deprecate_warning()
        return self.__class__(value=(self.value() + other.value()))

    def __sub__(self, other):
        """
        Allows subbing two records together
        to produce a brand new record.

        ** Note this does reverse subtraction because
            of the inverted nature of latency (lower is better)
        """

        self._print_deprecate_warning()
        return self.__class__(value=(other.value() - self.value()))

    @classmethod
    def _print_deprecate_warning(cls):
        if not cls._printed_deprecate_warning:
            logging.warning(
                "`perf_latency` is deprecated, use `perf_latency_p99` instead.")
            cls._printed_deprecate_warning = True
