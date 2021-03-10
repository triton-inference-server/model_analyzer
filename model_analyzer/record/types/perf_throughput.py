# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

from model_analyzer.record.record import Record


@total_ordering
class PerfThroughput(Record):
    """
    A record for perf_analyzer
    metric 'Throughput'
    """

    tag = "perf_throughput"

    def __init__(self, value, timestamp=0):
        """
        Parameters
        ----------
        value : float
            The throughput from the perf analyzer output
        timestamp : float
            Elapsed time from start of program
        """

        super().__init__(value, timestamp)

    @staticmethod
    def header(aggregation_tag=False):
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

        return "Throughput (infer/sec)"

    def __eq__(self, other):
        """
        Allows checking for
        equality between two records
        """

        return self.value() == other.value()

    def __lt__(self, other):
        """
        Allows checking if 
        this record is less than 
        the other
        """

        return self.value() < other.value()

    def __add__(self, other):
        """
        Allows adding two records together
        to produce a brand new record.
        """

        return PerfThroughput(value=(self.value() + other.value()))

    def __sub__(self, other):
        """
        Allows subtracting two records together
        to produce a brand new record.
        """

        return PerfThroughput(value=(self.value() - other.value()))
