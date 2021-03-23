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

from subprocess import Popen, STDOUT, PIPE
import logging
import psutil

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from model_analyzer.record.types.perf_latency import PerfLatency
from model_analyzer.record.types.perf_throughput import PerfThroughput

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

    def __init__(self, path, config, timeout, max_cpu_util):
        """
        Parameters
        ----------
        path : full path to the perf_analyzer
                executable
        config : PerfAnalyzerConfig
            keys are names of arguments to perf_analyzer,
            values are their values.
        timeout : int
            Maximum number of seconds that perf_analyzer
            will wait until the execution is complete.
        max_cpu_util : float
            Maximum CPU utilization allowed for perf_analyzer
        """

        self.bin_path = path
        self._config = config
        self._timeout = timeout
        self._output = None
        self._perf_records = None
        self._max_cpu_util = max_cpu_util

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
                process_killed = False

                process = Popen(cmd,
                                start_new_session=True,
                                stdout=PIPE,
                                stderr=STDOUT,
                                encoding='utf-8')
                current_timeout = self._timeout
                process_util = psutil.Process(process.pid)

                # Convert to miliseconds
                interval_sleep_time = self._config['measurement-interval'] // 1000
                while current_timeout > 0:
                    if process.poll() is not None:
                        self._output = process.stdout.read()
                        break

                    # perf_analyzer using too much CPU?
                    cpu_util = process_util.cpu_percent(interval_sleep_time)
                    if cpu_util > self._max_cpu_util:
                        logging.info(f'perf_analyzer used significant amount of CPU resources ({cpu_util}%), killing perf_analyzer...')
                        self._output = process.stdout.read()
                        process.kill()

                        # Failure
                        return 1

                    current_timeout -= interval_sleep_time
                else:
                    logging.info('perf_analyzer took very long to exit, killing perf_analyzer...')
                    process.kill()

                    # Failure
                    return 1

                if process_killed:
                    continue

                if process.returncode != 0:
                    if self._output.find("Please use a larger time window.") > 0:
                        self._config['measurement-interval'] += INTERVAL_DELTA
                        logger.info(
                            "perf_analyzer's measurement window is too small, "
                            f"increased to {self._config['measurement-interval']} ms."
                        )
                    else:
                        logging.info(
                            f"Running perf_analyzer {cmd} failed with"
                            f" exit status {process.returncode} : {self._output}")
                        return 1
                else:
                    self._parse_output()
                    break
            else:
                logging.info(
                    f"Ran perf_analyzer {MAX_INTERVAL_CHANGES} times, "
                    "but no valid requests recorded in max time interval"
                    f" of {self._config['measurement-interval']} ")
                return 1

        return 0

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
                self._perf_records.append(PerfThroughput(value=throughput))

            # Get first word and first word after 'latency:'
            elif 'p99 latency:' in line:
                latency_tags = line.split(' latency: ')

                # Convert value to ms from us
                latency = float(latency_tags[1].split()[0]) / 1e3
                self._perf_records.append(PerfLatency(value=latency))

        if not self._perf_records:
            raise TritonModelAnalyzerException(
                'perf_analyzer output was not as expected.')
