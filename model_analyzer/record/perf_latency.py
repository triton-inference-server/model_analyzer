# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import time

from .record import Record
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class PerfLatency(Record):
    """
    A record for perf_analyzer
    metric 'Avg Latency'
    """

    def __init__(self, perf_output, timestamp=0):
        """
        Parameters
        ----------
        perf_output : str
            The stdout from the perf_analyzer
        timestamp : float
            Elapsed time from start of program
        """

        perf_out_lines = perf_output.split('\n')
        for line in perf_out_lines[:-3]:
            # Get first word and first word after 'latency:'
            if 'latency:' in line:
                latency_tags = line.split(' latency: ')
                latency = float(latency_tags[1].split()[0])

        super().__init__(latency, timestamp)

    @staticmethod
    def header():
        """
        Returns
        -------
        str
            The full name of the
            metric.
        """

        return "Average Latency"
