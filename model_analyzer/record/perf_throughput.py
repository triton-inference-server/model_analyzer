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


class PerfThroughput(Record):
    """
    A record for perf_analyzer
    metric 'Throughput'
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
            # Get first word after Throughput
            if 'Throughput:' in line:
                throughput = float(line.split()[1])
                break
        else:
            raise TritonModelAnalyzerException(
                'perf_analyzer output was not as expected.')

        super().__init__(throughput, timestamp)

    @staticmethod
    def header():
        """
        Returns
        -------
        str
            The full name of the
            metric.
        """

        return "Throughput(infer/sec)"
