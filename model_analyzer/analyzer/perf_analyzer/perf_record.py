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

from model_analyzer.record.record import Record
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class PerfRecord(Record):
    """
    A record for perf_analyzer
    metrics
    """

    def __init__(self, perf_out):
        """
        Parameters
        ----------
        perf_out : str
            The raw stdout string from
            a run of the perf_analyzer

        Raises
        ------
        TritonModelAnalyzerException
            If there is a problem with the output format
            from the perf_analyzer output.
        """
        self._data = {}

        try:
            self._parse_perf_output(perf_out)
        except Exception as e:
            raise TritonModelAnalyzerException(e)

    def _parse_perf_output(self, perf_out):
        """
        Extracts required metrics from
        perf_analyzer output and sets
        properties for this class

        Parameters
        ----------
        out : str
            The raw output from the subprocess
            call to perf_analyzer
        """
        # Split output into lines and ignore last 3 lines
        perf_out_lines = perf_out.split('\n')
        for line in perf_out_lines[:-3]:
            # Last word in line with Batch and concurrency
            if 'Batch' in line:
                self._data['Batch size'] = int(line.split()[-1])
            elif 'concurrency:' in line:
                self._data['Concurrency'] = int(line.split()[-1])
            # Third word in measurement window line
            elif 'window:' in line:
                self._data['Measurement window'] = int(line.split()[2])
            # Get first and last word in lines with count
            elif 'count:' in line:
                [key, val] = line.split(' count: ')
                self._data[key.strip() + ' count'] = int(val)
            # Get first word and first word after 'latency:' in lines with
            # latency
            elif 'latency:' in line:
                latency_tags = line.split(' latency: ')
                self._data[latency_tags[0].strip(
                ) + ' latency'] = int(latency_tags[1].split()[0])
            # Get first word after Throughput
            elif 'Throughput:' in line:
                self._data['Throughput'] = float(line.split()[1])

    def __str__(self):
        """
        Prints a PerfRecord
        """

        out_str = f"*** Measurement Settings ***\n"
        out_str += f"  Batch size: {self._data['Batch size']}\n"
        out_str += f"  Measurement window: {self._data['Measurement window']} msec\n\n"
        out_str += f"Request concurrency: {self._data['Concurrency']}\n"
        out_str += f"  Client:\n"
        out_str += f"    Request count: {self._data['Request count']}\n"
        out_str += f"    Throughput: {self._data['Throughput']} infer/sec\n"
        out_str += f"    Avg latency: {self._data['Avg latency']} usec\n"
        out_str += f"    p50 latency: {self._data['p50 latency']} usec\n"
        out_str += f"    p90 latency: {self._data['p90 latency']} usec\n"
        out_str += f"    p95 latency: {self._data['p95 latency']} usec\n"
        out_str += f"    p99 latency: {self._data['p99 latency']} usec\n"
        out_str += f"  Server:\n"
        out_str += f"    Inference count: {self._data['Inference count']}\n"
        out_str += f"    Execution count: {self._data['Execution count']}\n"
        out_str += f"    Successful request count: {self._data['Successful request count']}\n"
        out_str += f"    Avg request latency: {self._data['Avg request latency']} usec\n"

        return out_str

    def to_dict(self):
        """
        Returns
        -------
        List
            The names of the metrics captured
            in this perf record.
        """
        return self._data
