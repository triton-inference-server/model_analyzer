#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.record.types.inter_token_latency_base import InterTokenLatencyBase


@total_ordering
class InterTokenLatencyP95(InterTokenLatencyBase):
    """
    A record for perf_analyzer Inter token latency metric
    """

    tag = "inter_token_latency_p95"

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

        return "p95 Inter Token Latency (ms)"
