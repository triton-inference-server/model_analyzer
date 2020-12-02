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

from itertools import product
from subprocess import check_output, CalledProcessError, STDOUT

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.record.perf_latency import PerfLatency
from model_analyzer.record.perf_throughput import PerfThroughput


class PerfAnalyzer:
    """
    This class provides an interface for running workloads
    with perf_analyzer.
    """

    # The metrics that PerfAnalyzer can collect
    perf_metrics = [PerfLatency, PerfThroughput]

    def __init__(self, path, config):
        """
        Parameters
        ----------
        path : full path to the perf_analyzer
                executable
        config : PerfAnalyzerConfig
            keys are names of arguments to perf_analyzer,
            values are their values.
        """

        self.bin_path = path
        self._config = config
        self._output = None

    def run(self, tags):
        """
        Runs the perf analyzer with the
        intialized configuration
        Parameters
        ----------
        tags : List of Record types
            The list of record types to parse from
            Perf Analyzer

        Returns
        -------
        List of Records
            List of the metrics obtained from this
            run of perf_analyzer

        Raises
        ------
        TritonModelAnalyzerException
            If subprocess throws CalledProcessError
        """

        if tags:
            # Synchronously start and finish run
            cmd = [self.bin_path]
            cmd += self._config.to_cli_string().replace('=', ' ').split()
            try:
                self._output = check_output(cmd,
                                            start_new_session=True,
                                            stderr=STDOUT,
                                            encoding='utf-8')
            except CalledProcessError as e:
                raise TritonModelAnalyzerException(
                    f"Running perf_analyzer with {e.cmd} failed with exit status {e.returncode} : {e.output}"
                )
        return [metric(self._output) for metric in tags]

    def output(self):
        """
        Returns
        -------
        The stdout output of the
        last perf_analyzer run
        """

        if self._output:
            return self._output
        raise TritonModelAnalyzerException(
            "Attempted to get perf_analyzer output"
            "without calling run first.")
