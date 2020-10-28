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
    def __init__(self, config):
        """
        Parameters
        ----------
        config : PerfAnalyzerConfig
            keys are names of arguments to perf_analyzer,
            values are their values.
        """
        self._config = config
        self._outputs = []

    def run(self):
        """
        Runs the perf analyzer with the
        intialized configuration

        Returns
        -------
        List of Records for metrics
        that are generated by the perf analyzer

        Raises
        ------
        TritonModelAnalyzerException
            If subprocess throws CalledProcessError
        """
        cmd = ['perf_analyzer']
        cmd += self._config.to_cli_string().replace('=', ' ').split()

        # Synchronously start and finish run
        try:
            out = check_output(cmd, stderr=STDOUT, encoding='utf-8')
        except CalledProcessError as e:
            raise TritonModelAnalyzerException(
                f"perf analyzer returned with exit"
                " status {e.returncode} : {e.output}")

        return self._parse_perf_output(out)

    def _parse_perf_output(self, perf_out):
        """
        Extracts required metrics from
        perf_analyzer output and sets
        properties for this class

        Parameters
        ----------
        perf_out : str
            The raw output from the subprocess
            call to perf_analyzer
        """
        data = {}
        # Split output into lines and ignore last 3 lines
        perf_out_lines = perf_out.split('\n')
        for line in perf_out_lines[:-3]:
            # Get first word and first word after 'latency:' in lines with
            # latency
            if 'latency:' in line:
                latency_tags = line.split(' latency: ')
                data[latency_tags[0].strip() + ' latency'] = float(
                    latency_tags[1].split()[0])
            # Get first word after Throughput
            elif 'Throughput:' in line:
                data['Throughput'] = float(line.split()[1])

        # Create and return the required records
        throughput_record = PerfThroughput(
            data['Throughput']) if ('Throughput' in data) else None
        latency_record = PerfLatency(
            data['Avg latency']) if ('Avg latency' in data) else None
        return (throughput_record, latency_record)
