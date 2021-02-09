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

from subprocess import check_output, CalledProcessError, STDOUT
import logging

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from model_analyzer.record.perf_latency import PerfLatency
from model_analyzer.record.perf_throughput import PerfThroughput

MAX_INTERVAL_CHANGES = 20
INTERVAL_DELTA = 1000

logger = logging.getLogger(__name__)


class PerfAnalyzer:
    """
    This class provides an interface for running workloads
    with perf_analyzer.
    """

    # The metrics that PerfAnalyzer can collect
    perf_metrics = {PerfLatency, PerfThroughput}

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
        self._perf_records = None

    def run(self, metrics):
        """
        Runs the perf analyzer with the
        intialized configuration
        Parameters
        ----------
        metrics : List of Record types
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

        if metrics:
            # Synchronously start and finish run
            for _ in range(MAX_INTERVAL_CHANGES):
                cmd = [self.bin_path]
                cmd += self._config.to_cli_string().replace('=', ' ').split()

                try:
                    self._output = check_output(cmd,
                                                start_new_session=True,
                                                stderr=STDOUT,
                                                encoding='utf-8')
                    self._parse_output()
                    return
                except CalledProcessError as e:
                    if e.output.find("Please use a larger time window.") > 0:
                        self._config['measurement-interval'] += INTERVAL_DELTA
                        logger.info(
                            "perf_analyzer's measurement window is too small, "
                            f"increased to {self._config['measurement-interval']} ms."
                        )
                    else:
                        raise TritonModelAnalyzerException(
                            f"Running perf_analyzer with {e.cmd} failed with"
                            f" exit status {e.returncode} : {e.output}")

            raise TritonModelAnalyzerException(
                f"Ran perf_analyzer {MAX_INTERVAL_CHANGES} times, "
                "but no valid requests recorded in max time interval"
                f" of {self._config['measurement-interval']} ")

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

    def get_records(self):
        """
        Returns
        -------
        The stdout output of the
        last perf_analyzer run
        """

        if self._perf_records:
            return self._perf_records
        raise TritonModelAnalyzerException(
            "Attempted to get perf_analyzer resultss"
            "without calling run first.")

    def _parse_output(self):
        """
        Extract metrics from the output of
        the perf_analyzer
        """

        self._perf_records = []
        perf_out_lines = self._output.split('\n')
        for line in perf_out_lines[:-3]:
            # Get first word after Throughput
            if 'Throughput:' in line:
                throughput = float(line.split()[1])
                self._perf_records.append(
                    PerfThroughput(throughput=throughput))

            # Get first word and first word after 'latency:'
            elif 'latency:' in line:
                latency_tags = line.split(' latency: ')
                latency = float(latency_tags[1].split()[0])
                self._perf_records.append(PerfLatency(latency=latency))

        if not self._perf_records:
            raise TritonModelAnalyzerException(
                'perf_analyzer output was not as expected.')
