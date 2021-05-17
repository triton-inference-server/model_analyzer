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

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from model_analyzer.record.types.perf_latency import PerfLatency
from model_analyzer.record.types.perf_throughput import PerfThroughput
from model_analyzer.record.types.perf_client_response_wait \
    import PerfClientResponseWait
from model_analyzer.record.types.perf_client_send_recv \
    import PerfClientSendRecv
from model_analyzer.record.types.perf_server_queue \
    import PerfServerQueue
from model_analyzer.record.types.perf_server_compute_input \
    import PerfServerComputeInput
from model_analyzer.record.types.perf_server_compute_infer \
    import PerfServerComputeInfer
from model_analyzer.record.types.perf_server_compute_output \
    import PerfServerComputeOutput

from model_analyzer.constants import INTERVAL_SLEEP_TIME, MAX_INTERVAL_CHANGES, MEASUREMENT_REQUEST_COUNT_STEP, MEASUREMENT_WINDOW_STEP, PERF_ANALYZER_MEASUREMENT_REQUEST_COUNT, PERF_ANALYZER_MEASUREMENT_WINDOW

from subprocess import Popen, STDOUT, PIPE
import logging
import psutil
import re

logger = logging.getLogger(__name__)


class PerfAnalyzer:
    """
    This class provides an interface for running workloads
    with perf_analyzer.
    """

    # The metrics that PerfAnalyzer can collect
    perf_metrics = {
        PerfLatency: "_parse_perf_latency",
        PerfThroughput: "_parse_perf_throughput",
        PerfClientSendRecv: "_parse_perf_client_send_recv",
        PerfClientResponseWait: "_parse_perf_client_response_wait",
        PerfServerQueue: "_parse_perf_server_queue",
        PerfServerComputeInfer: "_parse_perf_server_compute_infer",
        PerfServerComputeInput: "_parse_perf_server_compute_input",
        PerfServerComputeOutput: "_parse_perf_server_compute_output"
    }

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

                process = Popen(cmd,
                                start_new_session=True,
                                stdout=PIPE,
                                stderr=STDOUT,
                                encoding='utf-8')
                current_timeout = self._timeout
                process_util = psutil.Process(process.pid)

                while current_timeout > 0:
                    if process.poll() is not None:
                        self._output = process.stdout.read()
                        break

                    # perf_analyzer using too much CPU?
                    cpu_util = process_util.cpu_percent(INTERVAL_SLEEP_TIME)
                    if cpu_util > self._max_cpu_util:
                        logging.info(
                            f'perf_analyzer used significant amount of CPU resources ({cpu_util}%), killing perf_analyzer...'
                        )
                        self._output = process.stdout.read()
                        process.kill()

                        # Failure
                        return 1

                    current_timeout -= INTERVAL_SLEEP_TIME
                else:
                    logging.info(
                        'perf_analyzer took very long to exit, killing perf_analyzer...'
                    )
                    process.kill()

                    # Failure
                    return 1

                if process.returncode != 0:
                    if self._output.find(
                            "Failed to obtain stable measurement"
                    ) or self._output.find(
                            "Please use a larger time window") != -1:
                        if self._config['measurement-mode'] == 'time_windows':
                            if self._config['measurement-interval'] is None:
                                self._config[
                                    'measurement-interval'] = PERF_ANALYZER_MEASUREMENT_WINDOW + MEASUREMENT_WINDOW_STEP
                            else:
                                self._config['measurement-interval'] = int(
                                    self._config['measurement-interval']
                                ) + MEASUREMENT_WINDOW_STEP
                            logger.info(
                                "perf_analyzer's measurement window is too small, "
                                f"increased to {self._config['measurement-interval']} ms."
                            )
                        elif self._config[
                                'measurement-mode'] is None or self._config[
                                    'measurement-mode'] == 'count_windows':
                            if self._config[
                                    'measurement-request-count'] is None:
                                self._config[
                                    'measurement-request-count'] = PERF_ANALYZER_MEASUREMENT_REQUEST_COUNT + MEASUREMENT_REQUEST_COUNT_STEP
                            else:
                                self._config[
                                    'measurement-request-count'] = MEASUREMENT_REQUEST_COUNT_STEP + int(
                                        self.
                                        _config['measurement-request-count'])
                            logger.info(
                                "perf_analyzer's request count is small, "
                                f"increased to {self._config['measurement-request-count']}."
                            )
                    else:
                        logging.info(
                            f"Running perf_analyzer {cmd} failed with"
                            f" exit status {process.returncode} : {self._output}"
                        )
                        return 1
                else:
                    self._parse_output(metrics)
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

    def _parse_output(self, metrics):
        """
        Extract metrics from the output of
        the perf_analyzer
        """

        self._perf_records = []
        client_section_start = self._output.find('Client:')
        server_section_start = self._output.find('Server:',
                                                 client_section_start)
        server_section_end = self._output.find('Inferences/Second vs. Client',
                                               server_section_start)

        client_section = self._output[
            client_section_start:server_section_start].strip()
        server_section = self._output[
            server_section_start:server_section_end].strip()

        server_perf_metrics = {
            PerfServerQueue, PerfServerComputeInput, PerfServerComputeInfer,
            PerfServerComputeOutput
        }

        # Parse client values
        for metric in metrics:
            if metric not in self.perf_metrics:
                raise TritonModelAnalyzerException(
                    f"Perf metric : {metric} not found or supported.")
            if metric in server_perf_metrics:
                parse_func = getattr(self, self.perf_metrics[metric])
                self._perf_records.append(parse_func(server_section))
            else:
                parse_func = getattr(self, self.perf_metrics[metric])
                self._perf_records.append(parse_func(client_section))

        if not self._perf_records:
            raise TritonModelAnalyzerException(
                'perf_analyzer output was not as expected.')

    def _parse_perf_client_send_recv(self, section):
        """
        Parses Client send/recv values from the perf output
        """

        client_send_recv = None
        if self._config['protocol'] == 'http':
            # Http terminology
            client_send_recv = re.search('send/recv (\d+)', section)
        elif self._config['protocol'] == 'grpc':
            # Try gRPC related terminology
            client_send_recv = re.search('request/response (\d+)', section)
        if client_send_recv:
            client_send_recv = float(client_send_recv.group(1)) / 1e3
            return PerfClientSendRecv(value=client_send_recv)
        raise TritonModelAnalyzerException(
            'perf_analyzer output did not client send/recv time.')

    def _parse_perf_client_response_wait(self, section):
        """
        Parses Client response wait time (network + server send/recv)
        values from the perf output
        """
        client_response_wait = re.search('response wait (\d+)', section)
        if client_response_wait:
            client_response_wait = float(client_response_wait.group(1)) / 1e3
            return PerfClientResponseWait(value=client_response_wait)
        raise TritonModelAnalyzerException(
            'perf_analyzer output did not contain client response wait time.')

    def _parse_perf_throughput(self, section):
        """
        Parses throughput from the perf analyzer output
        """

        throughput = re.search('Throughput: (\d+\.\d+|\d+)', section)
        if throughput:
            throughput = float(throughput.group(1))
            return PerfThroughput(value=throughput)
        raise TritonModelAnalyzerException(
            'perf_analyzer output did not contain throughput.')

    def _parse_perf_latency(self, section):
        """
        Parses p99 latency from the perf output
        """

        p99_latency = re.search('p99 latency: (\d+\.\d+|\d+)', section)
        if p99_latency:
            p99_latency = float(p99_latency.group(1)) / 1e3
            return PerfLatency(value=p99_latency)
        raise TritonModelAnalyzerException(
            'perf_analyzer output did not contain p99 latency.')

    def _parse_perf_server_queue(self, section):
        """
        Parses serve queue time from the perf output
        """

        server_queue = re.search('queue (\d+) usec', section)
        if server_queue:
            server_queue = float(server_queue.group(1)) / 1e3
            return PerfServerQueue(value=server_queue)
        raise TritonModelAnalyzerException(
            'perf_analyzer output did not server queue time.')

    def _parse_perf_server_compute_input(self, section):
        """
        Parses server compute input time from the perf output
        """

        server_compute_input = re.search('compute input (\d+) usec', section)
        if server_compute_input:
            server_compute_input = float(server_compute_input.group(1)) / 1e3
            return PerfServerComputeInput(value=server_compute_input)
        raise TritonModelAnalyzerException(
            'perf_analyzer output did not server compute input time.')

    def _parse_perf_server_compute_infer(self, section):
        """
        Parses server compute infer time from the perf output
        """

        server_compute_infer = re.search('compute infer (\d+) usec', section)
        if server_compute_infer:
            server_compute_infer = float(server_compute_infer.group(1)) / 1e3
            return PerfServerComputeInfer(value=server_compute_infer)
        raise TritonModelAnalyzerException(
            'perf_analyzer output did not server compute infer time.')

    def _parse_perf_server_compute_output(self, section):
        """
        Parses server compute output time from the perf output
        """

        server_compute_output = re.search('compute output (\d+) usec', section)
        if server_compute_output:
            server_compute_output = float(server_compute_output.group(1)) / 1e3
            return PerfServerComputeOutput(value=server_compute_output)
        raise TritonModelAnalyzerException(
            'perf_analyzer output did not server compute output time.')
