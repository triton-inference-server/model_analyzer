#!/usr/bin/env python3

# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import csv
import glob
import logging
import os
import re
import signal
import tempfile
from csv import DictReader
from subprocess import STDOUT, Popen
from typing import Dict, List, Optional

import psutil

from model_analyzer.config.input.config_defaults import DEFAULT_MODEL_TYPE
from model_analyzer.constants import (
    GENAI_PERF_COLLATERAL,
    GENAI_PERF_CSV,
    INTERVAL_SLEEP_TIME,
    LOGGER_NAME,
    MEASUREMENT_REQUEST_COUNT_STEP,
    MEASUREMENT_WINDOW_STEP,
    PERF_ANALYZER_MEASUREMENT_WINDOW,
    PERF_ANALYZER_MINIMUM_REQUEST_COUNT,
)
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.record.record import Record
from model_analyzer.record.types.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.types.gpu_power_usage import GPUPowerUsage
from model_analyzer.record.types.gpu_used_memory import GPUUsedMemory
from model_analyzer.record.types.gpu_utilization import GPUUtilization
from model_analyzer.record.types.inter_token_latency_avg import InterTokenLatencyAvg
from model_analyzer.record.types.inter_token_latency_max import InterTokenLatencyMax
from model_analyzer.record.types.inter_token_latency_min import InterTokenLatencyMin
from model_analyzer.record.types.inter_token_latency_p25 import InterTokenLatencyP25
from model_analyzer.record.types.inter_token_latency_p50 import InterTokenLatencyP50
from model_analyzer.record.types.inter_token_latency_p75 import InterTokenLatencyP75
from model_analyzer.record.types.inter_token_latency_p90 import InterTokenLatencyP90
from model_analyzer.record.types.inter_token_latency_p95 import InterTokenLatencyP95
from model_analyzer.record.types.inter_token_latency_p99 import InterTokenLatencyP99
from model_analyzer.record.types.output_token_throughput import OutputTokenThroughput
from model_analyzer.record.types.perf_client_response_wait import PerfClientResponseWait
from model_analyzer.record.types.perf_client_send_recv import PerfClientSendRecv
from model_analyzer.record.types.perf_latency_avg import PerfLatencyAvg
from model_analyzer.record.types.perf_latency_p90 import PerfLatencyP90
from model_analyzer.record.types.perf_latency_p95 import PerfLatencyP95
from model_analyzer.record.types.perf_latency_p99 import PerfLatencyP99
from model_analyzer.record.types.perf_server_compute_infer import PerfServerComputeInfer
from model_analyzer.record.types.perf_server_compute_input import PerfServerComputeInput
from model_analyzer.record.types.perf_server_compute_output import (
    PerfServerComputeOutput,
)
from model_analyzer.record.types.perf_server_queue import PerfServerQueue
from model_analyzer.record.types.perf_throughput import PerfThroughput
from model_analyzer.record.types.time_to_first_token_avg import TimeToFirstTokenAvg
from model_analyzer.record.types.time_to_first_token_max import TimeToFirstTokenMax
from model_analyzer.record.types.time_to_first_token_min import TimeToFirstTokenMin
from model_analyzer.record.types.time_to_first_token_p25 import TimeToFirstTokenP25
from model_analyzer.record.types.time_to_first_token_p50 import TimeToFirstTokenP50
from model_analyzer.record.types.time_to_first_token_p75 import TimeToFirstTokenP75
from model_analyzer.record.types.time_to_first_token_p90 import TimeToFirstTokenP90
from model_analyzer.record.types.time_to_first_token_p95 import TimeToFirstTokenP95
from model_analyzer.record.types.time_to_first_token_p99 import TimeToFirstTokenP99

logger = logging.getLogger(LOGGER_NAME)


class PerfAnalyzer:
    """
    This class provides an interface for running workloads
    with perf_analyzer.
    """

    GPU_METRIC_UUID = 0
    GPU_METRIC_VALUE = 1

    # yapf: disable
    PA_SUCCESS, PA_FAIL, PA_RETRY = 0, 1, 2

    METRIC_TAG,                        CSV_STRING,             RECORD_CLASS,             REDUCTION_FACTOR = 0, 1, 2, 3
    perf_metric_table = [
        ["perf_latency_avg",           "Avg latency",           PerfLatencyAvg,          "1000"],
        ["perf_latency_p90",           "p90 latency",           PerfLatencyP90,          "1000"],
        ["perf_latency_p95",           "p95 latency",           PerfLatencyP95,          "1000"],
        ["perf_latency_p99",           "p99 latency",           PerfLatencyP99,          "1000"],
        ["perf_throughput",            "Inferences/Second",     PerfThroughput,             "1"],
        ["perf_client_send_recv",      "request/response",      PerfClientSendRecv,      "1000"],
        ["perf_client_send_recv",      "send/recv",             PerfClientSendRecv,      "1000"],
        ["perf_client_response_wait",  "response wait",         PerfClientResponseWait,  "1000"],
        ["perf_server_queue",          "Server Queue",          PerfServerQueue,         "1000"],
        ["perf_server_compute_infer",  "Server Compute Infer",  PerfServerComputeInfer,  "1000"],
        ["perf_server_compute_input",  "Server Compute Input",  PerfServerComputeInput,  "1000"],
        ["perf_server_compute_output", "Server Compute Output", PerfServerComputeOutput, "1000"]
    ]

    gpu_metric_table = [
        ["gpu_utilization",            "Avg GPU Utilization",   GPUUtilization,          "0.01"],
        ["gpu_power_usage",            "Avg GPU Power Usage",   GPUPowerUsage,              "1"],
        ["gpu_used_memory",            "Max GPU Memory Usage",  GPUUsedMemory,        "1000000"],
        ["gpu_free_memory",            "Total GPU Memory",      GPUFreeMemory,        "1000000"]
    ]

    llm_metric_table = [
        ["time_to_first_token_avg", "Time To First Token (ns) avg",          TimeToFirstTokenAvg, "1000"],
        ["time_to_first_token_min", "Time To First Token (ns) min",          TimeToFirstTokenMin, "1000"],
        ["time_to_first_token_max", "Time To First Token (ns) max",          TimeToFirstTokenMax, "1000"],
        ["time_to_first_token_p99", "Time To First Token (ns) p99",          TimeToFirstTokenP99, "1000"],
        ["time_to_first_token_p95", "Time To First Token (ns) p95",          TimeToFirstTokenP95, "1000"],
        ["time_to_first_token_p90", "Time To First Token (ns) p90",          TimeToFirstTokenP90, "1000"],
        ["time_to_first_token_p75", "Time To First Token (ns) p75",          TimeToFirstTokenP75, "1000"],
        ["time_to_first_token_p50", "Time To First Token (ns) p50",          TimeToFirstTokenP50, "1000"],
        ["time_to_first_token_p25", "Time To First Token (ns) p25",          TimeToFirstTokenP25, "1000"],
        ["inter_token_latency_avg", "Inter Token Latency (ns) avg",          InterTokenLatencyAvg, "1000"],
        ["inter_token_latency_min", "Inter Token Latency (ns) min",          InterTokenLatencyMin, "1000"],
        ["inter_token_latency_max", "Inter Token Latency (ns) max",          InterTokenLatencyMax, "1000"],
        ["inter_token_latency_p99", "Inter Token Latency (ns) p99",          InterTokenLatencyP99, "1000"],
        ["inter_token_latency_p95", "Inter Token Latency (ns) p95",          InterTokenLatencyP95, "1000"],
        ["inter_token_latency_p90", "Inter Token Latency (ns) p90",          InterTokenLatencyP90, "1000"],
        ["inter_token_latency_p75", "Inter Token Latency (ns) p75",          InterTokenLatencyP75, "1000"],
        ["inter_token_latency_p50", "Inter Token Latency (ns) p50",          InterTokenLatencyP50, "1000"],
        ["inter_token_latency_p25", "Inter Token Latency (ns) p25",          InterTokenLatencyP25, "1000"],
        ["output_token_throughput", "Output Token Throughput (per sec) avg", OutputTokenThroughput, "1"]
    ]
    # yapf: enable

    @staticmethod
    def get_perf_metrics():
        perf_metrics = [
            perf_metric[PerfAnalyzer.RECORD_CLASS]
            for perf_metric in PerfAnalyzer.perf_metric_table
        ]
        return perf_metrics

    @staticmethod
    def get_gpu_metrics():
        gpu_metrics = [
            gpu_metric[PerfAnalyzer.RECORD_CLASS]
            for gpu_metric in PerfAnalyzer.gpu_metric_table
        ]
        return gpu_metrics

    @staticmethod
    def get_llm_metrics():
        llm_metrics = [
            llm_metric[PerfAnalyzer.RECORD_CLASS]
            for llm_metric in PerfAnalyzer.llm_metric_table
        ]
        return llm_metrics

    def __init__(
        self,
        path,
        config,
        max_retries,
        timeout,
        max_cpu_util,
        model_type=DEFAULT_MODEL_TYPE,
    ):
        """
        Parameters
        ----------
        path : full path to the perf_analyzer
                executable
        config : RunConfig
            The RunConfig with information on what to execute
        max_retries: int
            Maximum number of times perf_analyzer adjusts parameters
            in an attempt to profile a model.
        timeout : int
            Maximum number of seconds that perf_analyzer
            will wait until the execution is complete.
        max_cpu_util : float
            Maximum CPU utilization allowed for perf_analyzer
        """

        self.bin_path = path
        self._config = config
        self._max_retries = max_retries
        self._timeout = timeout
        self._output = ""
        self._perf_records = {}
        self._llm_records = {}
        self._gpu_records = []
        self._max_cpu_util = max_cpu_util
        self._model_type = model_type

    def run(self, metrics, env=None):
        """
        Runs the perf analyzer with the
        initialized configuration
        Parameters
        ----------
        metrics : List of Record types
            The list of record types to parse from
            Perf Analyzer

        env: dict
            Environment variables to set for perf_analyzer run

        Returns
        -------
        Dict
            Dict of Model to List of Records obtained from this
            run of perf_analyzer

        Raises
        ------
        TritonModelAnalyzerException
            If subprocess throws CalledProcessError
        """

        if metrics:
            # Synchronously start and finish run
            for _ in range(self._max_retries):
                status = self._execute_pa(env)

                if status == self.PA_FAIL:
                    return status
                elif status == self.PA_SUCCESS:
                    self._parse_outputs(metrics)
                    break
                elif status == self.PA_RETRY:
                    continue
                else:
                    raise TritonModelAnalyzerException(f"Unexpected PA return {status}")

            else:
                logger.info(
                    f"Ran perf_analyzer {self._max_retries} times, "
                    "but no valid requests recorded"
                )
                return self.PA_FAIL

        return self.PA_SUCCESS

    def get_perf_records(self):
        """
        Returns
        -------
        The perf records from the last perf_analyzer run
        """

        if self._perf_records:
            return self._perf_records
        raise TritonModelAnalyzerException(
            "Attempted to get perf_analyzer results without calling run first."
        )

    def get_llm_records(self):
        """
        Returns
        -------
        The LLM records from the last perf_analyzer run
        """

        if self._llm_records:
            return self._llm_records
        raise TritonModelAnalyzerException(
            "Attempted to get perf_analyzer results without calling run first."
        )

    def get_gpu_records(self):
        """
        Returns
        -------
        The gpu records from the last perf_analyzer run
        """

        return self._gpu_records

    def output(self):
        """
        Returns
        -------
        The stdout output of the
        last perf_analyzer run
        """

        if not self._output:
            logger.info("perf_analyzer did not produce any output.")
        return self._output

    def get_cmd(self):
        """
        Returns a string of the command to run
        """
        return " ".join(self._get_cmd())

    def _execute_pa(self, env):
        cmd = self._get_cmd()
        logger.debug(f"Running {cmd}")
        perf_analyzer_env = self._create_env(env)

        process = self._create_process(cmd, perf_analyzer_env)
        status = self._resolve_process(process)

        return status

    def _get_cmd(self):
        if self._is_multi_model():
            cmd = ["mpiexec", "--allow-run-as-root", "--tag-output"]
            for index in range(len(self._config.model_run_configs())):
                if index:
                    cmd += [":"]
                cmd += ["-n", "1"]
                cmd += self._get_single_model_cmd(index)
        else:
            cmd = self._get_single_model_cmd(0)
        return cmd

    def _get_single_model_cmd(self, index):
        if self._model_type == "LLM":
            cmd = ["genai-perf", "-m", self._config.models_name()]
            cmd += self._get_genai_perf_cli_command(index).replace("=", " ").split()
            cmd += ["--"]
            cmd += (
                self._get_pa_cli_command(index, exclude_model_name=True)
                .replace("=", " ")
                .split()
            )
        else:
            cmd = [self.bin_path]
            if self._is_multi_model():
                cmd += ["--enable-mpi"]
            cmd += self._get_pa_cli_command(index).replace("=", " ").split()

        return cmd

    def _get_pa_cli_command(self, index, exclude_model_name=False):
        return (
            self._config.model_run_configs()[index]
            .perf_config()
            .to_cli_string(exclude_model_name)
        )

    def _get_genai_perf_cli_command(self, index):
        return self._config.genai_perf_config().to_cli_string()

    def _create_env(self, env):
        perf_analyzer_env = os.environ.copy()

        if env:
            # Filter env variables that use env lookups
            for variable, value in env.items():
                if value.find("$") == -1:
                    perf_analyzer_env[variable] = value
                else:
                    # Collect the ones that need lookups to give to the shell
                    perf_analyzer_env[variable] = os.path.expandvars(value)

        return perf_analyzer_env

    def _create_process(self, cmd, perf_analyzer_env):
        self._cmd_log = tempfile.NamedTemporaryFile()
        try:
            process = Popen(
                cmd,
                start_new_session=True,
                stdout=self._cmd_log,
                stderr=STDOUT,
                encoding="utf-8",
                env=perf_analyzer_env,
            )
        except FileNotFoundError as e:
            raise TritonModelAnalyzerException(f"perf_analyzer binary not found : {e}")
        return process

    def _resolve_process(self, process):
        if self._poll_perf_analyzer(process) == 1:
            return self.PA_FAIL

        if process.returncode > 0:
            if self._auto_adjust_parameters(process) == self.PA_FAIL:
                return self.PA_FAIL
            else:
                return self.PA_RETRY
        elif process.returncode < 0:
            logger.error(
                "perf_analyzer was terminated by signal: "
                f"{signal.Signals(abs(process.returncode)).name}"
            )
            return self.PA_FAIL

        return self.PA_SUCCESS

    def _poll_perf_analyzer(self, process):
        """
        Periodically poll the perf analyzer to get output
        or see if it is taking too much time or CPU resources
        """

        current_timeout = self._timeout
        process_util = psutil.Process(process.pid)

        while current_timeout > 0:
            if process.poll() is not None:
                self._output = self._get_process_output()
                break

            # perf_analyzer using too much CPU?
            cpu_util = process_util.cpu_percent(INTERVAL_SLEEP_TIME)
            if cpu_util > self._max_cpu_util:
                logger.info(
                    f"perf_analyzer used significant amount of CPU resources ({cpu_util}%), killing perf_analyzer"
                )
                self._output = self._get_process_output()
                process.kill()

                return self.PA_FAIL

            current_timeout -= INTERVAL_SLEEP_TIME
        else:
            logger.info("perf_analyzer took very long to exit, killing perf_analyzer")
            process.kill()

            return self.PA_FAIL

        return self.PA_SUCCESS

    def _get_process_output(self):
        self._cmd_log.seek(0)
        tmp_output = self._cmd_log.read()
        self._cmd_log.close()

        # PA has occasionally output non-UTF-8 bytes which would cause MA
        # to assert. In that case, just ignore the result instead of asserting
        result = ""
        try:
            result = tmp_output.decode("utf-8")
        except Exception:
            # Ignore the result on decode failed
            pass

        return result

    def _auto_adjust_parameters(self, process):
        """
        Attempt to update PA parameters based on the output
        """
        if (
            self._output.find("Failed to obtain stable measurement") != -1
            or self._output.find("Please use a larger time window") != -1
        ):
            per_rank_logs = self._split_output_per_rank()

            for index, log in enumerate(per_rank_logs):
                perf_config = self._config.model_run_configs()[index].perf_config()
                self._auto_adjust_parameters_for_perf_config(perf_config, log)

            return self.PA_SUCCESS
        else:
            clamped_output = self._output[:1000]
            logger.info(
                f"Running perf_analyzer failed with"
                f" exit status {process.returncode}:\n{clamped_output}"
            )
            return self.PA_FAIL

    def _auto_adjust_parameters_for_perf_config(self, perf_config, log):
        if (
            log.find("Failed to obtain stable measurement") != -1
            or log.find("Please use a larger time window") != -1
        ):
            if perf_config["measurement-mode"] == "time_windows":
                if perf_config["measurement-interval"] is None:
                    perf_config["measurement-interval"] = (
                        PERF_ANALYZER_MEASUREMENT_WINDOW + MEASUREMENT_WINDOW_STEP
                    )
                else:
                    perf_config["measurement-interval"] = (
                        int(perf_config["measurement-interval"])
                        + MEASUREMENT_WINDOW_STEP
                    )

                logger.info(
                    "perf_analyzer's measurement window is too small, "
                    f"increased to {perf_config['measurement-interval']} ms."
                )
            elif (
                perf_config["measurement-mode"] is None
                or perf_config["measurement-mode"] == "count_windows"
            ):
                if perf_config["measurement-request-count"] is None:
                    perf_config["measurement-request-count"] = (
                        PERF_ANALYZER_MINIMUM_REQUEST_COUNT
                        + MEASUREMENT_REQUEST_COUNT_STEP
                    )
                else:
                    perf_config["measurement-request-count"] = (
                        int(perf_config["measurement-request-count"])
                        + MEASUREMENT_REQUEST_COUNT_STEP
                    )

                logger.info(
                    "perf_analyzer's request count is too small, "
                    f"increased to {perf_config['measurement-request-count']}."
                )

    def _split_output_per_rank(self):
        if self._is_multi_model():
            outputs = ["" for mrc in self._config.model_run_configs()]
            for line in self._output.splitlines():
                # Example would find the '2': [1,2]<stdout>: fake output ***
                rank = re.search("^\[\d+,(\d+)\]", line)

                if rank:
                    index = int(rank.group(1))
                    outputs[index] += line + "\n"
            return outputs
        else:
            return [self._output]

    def _is_multi_model(self):
        """
        Returns true if the RunConfig provided to this class contains multiple perf_configs. Else False
        """
        return len(self._config.model_run_configs()) > 1

    def _parse_outputs(self, metrics):
        self._parse_generic_outputs(metrics)

        if self._model_type == "LLM":
            self._parse_llm_outputs(metrics)

    def _parse_generic_outputs(self, metrics):
        """
        Extract records from the Perf Analyzer run for each model
        """

        for perf_config in [
            mrc.perf_config() for mrc in self._config.model_run_configs()
        ]:
            logger.debug(
                f"Reading PA results from {perf_config['latency-report-file']}"
            )
            with open(perf_config["latency-report-file"], mode="r") as f:
                csv_reader = csv.DictReader(f, delimiter=",")

                for row in csv_reader:
                    self._perf_records[
                        perf_config["model-name"]
                    ] = self._extract_perf_records_from_row(metrics, row)
                    self._gpu_records = self._extract_gpu_records_from_row(metrics, row)

        for perf_config in [
            mrc.perf_config() for mrc in self._config.model_run_configs()
        ]:
            # Remove the latency file and all associated composing model latency files
            for f in glob.glob(f"*{perf_config['latency-report-file']}"):
                os.remove(f)

    def _parse_llm_outputs(self, metrics):
        """
        Extract records from the Perf Analyzer run for each model
        """

        perf_config = self._config.model_run_configs()[0].perf_config()

        logger.debug(f"Reading GENAI-PERF results from {GENAI_PERF_CSV}")
        with open(GENAI_PERF_CSV, mode="r") as f:
            csv_reader = list(csv.DictReader(f, delimiter=","))

            # See test_perf_analyzer::test_pa_llm_csv_output() for CSV output example
            self._llm_records[perf_config["model-name"]] = self._extract_llm_records(
                metrics, csv_reader
            )

            os.remove(GENAI_PERF_CSV)
            for filename in GENAI_PERF_COLLATERAL:
                os.remove(filename)

    def _extract_perf_records_from_row(
        self, requested_metrics: List[Record], row_metrics: Dict[str, str]
    ) -> List[Record]:
        perf_records: List[Record] = []
        for perf_metric in PerfAnalyzer.perf_metric_table:
            if self._is_metric_requested_and_in_row(
                perf_metric, requested_metrics, row_metrics
            ):
                value = float(row_metrics[str(perf_metric[PerfAnalyzer.CSV_STRING])])
                reduction_factor = float(
                    str(perf_metric[PerfAnalyzer.REDUCTION_FACTOR])
                )
                perf_value = value / reduction_factor

                perf_records.append(
                    perf_metric[PerfAnalyzer.RECORD_CLASS](perf_value)  # type: ignore
                )

        return perf_records

    def _extract_gpu_records_from_row(
        self, requested_metrics: List[Record], row_metrics: Dict[str, str]
    ) -> List[Record]:
        # GPU metrics have the following format: UUID0:value0;UUID1:value1;...
        gpu_records: List[Record] = []
        for gpu_metric in PerfAnalyzer.gpu_metric_table:
            if self._is_metric_requested_and_in_row(
                gpu_metric, requested_metrics, row_metrics
            ):
                gpu_metric_string = row_metrics[
                    str(gpu_metric[PerfAnalyzer.CSV_STRING])
                ]

                # Covers the case where PA didn't provide data
                if not gpu_metric_string:
                    continue

                # Needed because PA might terminate substring with a ;
                if gpu_metric_string and gpu_metric_string[-1] == ";":
                    gpu_metric_string = gpu_metric_string[:-1]

                gpu_metric_string_tuples = gpu_metric_string.split(";")

                for gpu_metric_string_tuple in gpu_metric_string_tuples:
                    gpu_metric_tuple = gpu_metric_string_tuple.split(":")

                    uuid = gpu_metric_tuple[PerfAnalyzer.GPU_METRIC_UUID]
                    tmp_value = float(gpu_metric_tuple[PerfAnalyzer.GPU_METRIC_VALUE])
                    reduction_factor = float(
                        str(gpu_metric[PerfAnalyzer.REDUCTION_FACTOR])
                    )
                    value = tmp_value / reduction_factor

                    record = gpu_metric[PerfAnalyzer.RECORD_CLASS](
                        value=value, device_uuid=uuid
                    )  # type: ignore

                    gpu_records.append(record)

        self._cleanup_gpu_records(gpu_records)
        return gpu_records

    def _extract_llm_records(
        self, requested_metrics: List[Record], csv_reader: DictReader
    ) -> List[Record]:
        llm_records: List[Record] = []

        for requested_metric in requested_metrics:
            new_llm_record = self._get_llm_record_from_csv(requested_metric, csv_reader)
            if new_llm_record:
                llm_records.append(new_llm_record)

        return llm_records

    def _get_llm_record_from_csv(
        self, requested_metric: Record, csv_reader: DictReader
    ) -> Optional[Record]:
        for row in csv_reader:
            for key, value in row.items():
                metric_string = f"{row['Metric']} {key}"
                llm_metric = self._find_corresponding_llm_metric_row(metric_string)

                if (
                    llm_metric
                    and llm_metric[PerfAnalyzer.METRIC_TAG] == requested_metric.tag
                ):
                    adjusted_value = float(value) / float(
                        llm_metric[PerfAnalyzer.REDUCTION_FACTOR]
                    )

                    llm_record = llm_metric[PerfAnalyzer.RECORD_CLASS](adjusted_value)  # type: ignore
                    return llm_record

        return None

    def _find_corresponding_llm_metric_row(self, metric_string: str) -> Optional[List]:
        for row in PerfAnalyzer.llm_metric_table:
            if metric_string == row[PerfAnalyzer.CSV_STRING]:
                return row

        return None

    def _cleanup_gpu_records(self, gpu_records):
        # Recalculate GPUFreeMemory by removing the value of the associated GPUUsedMemory
        # Remove any GPUFreeMemory records that don't have a matching GPUUsedMemory
        indexes_to_remove = []
        for i, record in enumerate(gpu_records):
            if type(record) == GPUFreeMemory:
                # Find matching UUID UsedMemory
                found = False
                for other_record in gpu_records:
                    if (
                        type(other_record) == GPUUsedMemory
                        and record.device_uuid() == other_record.device_uuid()
                    ):
                        found = True
                        record._value = record.value() - other_record.value()
                        break
                if not found:
                    indexes_to_remove.append(i)
        for i in reversed(indexes_to_remove):
            del gpu_records[i]

    def _is_metric_requested_and_in_row(
        self,
        metric: List[object],
        requested_metrics: List[Record],
        row_metrics: Dict[str, str],
    ) -> bool:
        tag_match = any(
            metric[PerfAnalyzer.METRIC_TAG] in requested_metric.tag
            for requested_metric in requested_metrics
        )

        return tag_match and metric[PerfAnalyzer.CSV_STRING] in row_metrics
