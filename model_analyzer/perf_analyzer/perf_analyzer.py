# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
                                            stderr=STDOUT,
                                            encoding='utf-8')
            except CalledProcessError as e:
                raise TritonModelAnalyzerException(
                    f"Running perf_analyzer with {e.cmd} failed with exit status {e.returncode}"
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
