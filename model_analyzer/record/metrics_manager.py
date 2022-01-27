# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .record_aggregator import RecordAggregator
from .record import RecordType
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.device.gpu_device_factory import GPUDeviceFactory
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from model_analyzer.monitor.cpu_monitor import CPUMonitor
from model_analyzer.monitor.dcgm.dcgm_monitor import DCGMMonitor
from model_analyzer.monitor.remote_monitor import RemoteMonitor
from model_analyzer.output.file_writer import FileWriter
from model_analyzer.perf_analyzer.perf_analyzer import PerfAnalyzer
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.result.measurement import Measurement

from collections import defaultdict
from prometheus_client.parser import text_string_to_metric_families
import numba
import requests
import logging
import os
import time

logger = logging.getLogger(LOGGER_NAME)


class MetricsManager:
    """
    This class handles the profiling
    categorization of metrics
    """

    metrics = [
        "perf_throughput", "perf_latency_avg", "perf_latency_p90",
        "perf_latency_p95", "perf_latency_p99", "perf_latency",
        "perf_client_response_wait", "perf_client_send_recv",
        "perf_server_queue", "perf_server_compute_input",
        "perf_server_compute_infer", "perf_server_compute_output",
        "gpu_used_memory", "gpu_free_memory", "gpu_utilization",
        "gpu_power_usage", "cpu_available_ram", "cpu_used_ram"
    ]

    def __init__(self, config, client, server, gpus, result_manager,
                 state_manager):
        """
        Parameters
        ----------
        config :ConfigCommandProfile
            The model analyzer's config
        client : TritonClient
            handle to the instance of Tritonclient to communicate with
            the server
        server : TritonServer
            Handle to the instance of Triton being used
        gpus: List of GPUDevices
            The gpus being used to profile
        result_manager : ResultManager
            instance that manages the result tables and
            adding results
        state_manager: AnalyzerStateManager
            manages the analyzer state
        """

        # Generate the output model repository path folder.
        self._output_model_repo_path = config.output_model_repository_path

        if len(config.profile_models) != len(
                set([model._model_name for model in config.profile_models])):
            raise TritonModelAnalyzerException(
                f"Duplicate model names detected: "
                f"{[model._model_name for model in config.profile_models]}")
        self._first_config_variant = {}

        self._config = config
        self._client = client
        self._server = server
        self._result_manager = result_manager
        self._state_manager = state_manager

        self._gpu_metrics, self._perf_metrics, self._cpu_metrics = self._categorize_metrics(
            self.metrics, self._config.collect_cpu_metrics)
        self._gpus = gpus
        self._init_state()

    def start_new_model(self):
        """ Indicate that profiling of a new model is starting """
        self._first_config_variant = {}

    def _init_state(self):
        """
        Sets MetricsManager object managed
        state variables in AnalyerState
        """

        gpu_info = self._state_manager.get_state_variable(
            'MetricsManager.gpu_info')

        if self._state_manager.starting_fresh_run() or gpu_info is None:
            gpu_info = {}

        for i in range(len(self._gpus)):
            if self._gpus[i].device_uuid() not in gpu_info:
                device_info = {}
                device = numba.cuda.list_devices()[i]
                device_info['name'] = device.name
                with device:
                    # convert bytes to GB
                    device_info['total_memory'] = numba.cuda.current_context(
                    ).get_memory_info().total
                gpu_info[self._gpus[i].device_uuid()] = device_info

        self._state_manager.set_state_variable('MetricsManager.gpus', gpu_info)

    @staticmethod
    def _categorize_metrics(metric_tags, collect_cpu_metrics=False):
        """
        Splits the metrics into groups based
        on how they are collected

        Returns
        -------
        (list,list,list)
            tuple of three lists (DCGM, PerfAnalyzer, CPU) metrics
        """

        gpu_metrics, perf_metrics, cpu_metrics = [], [], []
        # Separates metrics and objectives into related lists
        for metric in MetricsManager.get_metric_types(metric_tags):
            if metric in DCGMMonitor.model_analyzer_to_dcgm_field or metric in RemoteMonitor.gpu_metrics.values(
            ):
                gpu_metrics.append(metric)
            elif metric in PerfAnalyzer.perf_metrics:
                perf_metrics.append(metric)
            elif collect_cpu_metrics and (metric in CPUMonitor.cpu_metrics):
                cpu_metrics.append(metric)

        return gpu_metrics, perf_metrics, cpu_metrics

    def profile_server(self):
        """
        Runs the DCGM monitor on the triton server without the perf_analyzer
        Raises
        ------
        TritonModelAnalyzerException
        """

        cpu_only = (not numba.cuda.is_available())
        self._start_monitors(cpu_only=cpu_only)
        time.sleep(self._config.duration_seconds)
        if not cpu_only:
            server_gpu_metrics = self._get_gpu_inference_metrics()
            self._result_manager.add_server_data(data=server_gpu_metrics)
        self._destroy_monitors(cpu_only=cpu_only)

    def execute_run_config(self, run_config):
        """
        Executes the run config. Returns obtained measurement. Also sends 
        measurement to the result manager
        """

        # Create model variants
        self._create_model_variants(run_config)

        # If this run config was already run, do not run again, just get the measurement
        measurement = self._get_measurement_if_config_duplicate(run_config)
        if measurement:
            return measurement

        # Start server, and load model variants
        self._server.start(env=run_config.triton_environment())
        if not self._load_model_variants(run_config):
            self._server.stop()
            return

        # Profile various batch size and concurrency values.
        measurement = self.profile_model(run_config=run_config)

        self._server.stop()

        return measurement

    def _create_model_variants(self, run_config):
        """
        Creates and fills all model variant directories
        """

        for model_config in run_config.model_configs():
            self._create_model_variant(original_name=run_config.model_name(),
                                       variant_config=model_config)

    def _create_model_variant(self, original_name, variant_config):
        """
        Creates a directory for the model config variant in the output model
        repository and fills directory with config
        """

        variant_name = variant_config.get_field('name')
        if self._config.triton_launch_mode != 'remote':
            model_repository = self._config.model_repository

            original_model_dir = os.path.join(model_repository, original_name)
            new_model_dir = os.path.join(self._output_model_repo_path,
                                         variant_name)
            try:
                # Create the directory for the new model
                os.makedirs(new_model_dir, exist_ok=False)
                self._first_config_variant.setdefault(original_name, None)
                variant_config.write_config_to_file(
                    new_model_dir, original_model_dir,
                    self._first_config_variant[original_name])
                if self._first_config_variant[original_name] is None:
                    self._first_config_variant[original_name] = os.path.join(
                        self._output_model_repo_path, variant_name)
            except FileExistsError:
                pass

    def _load_model_variants(self, run_config):
        """
        Loads all model variants in the client
        """

        for model_config in run_config.model_configs():
            if not self._load_model_variant(variant_config=model_config):
                return False
        return True

    def _load_model_variant(self, variant_config):
        """
        Conditionally loads a model variant in the client
        """
        remote = self._config.triton_launch_mode == 'remote'
        c_api = self._config.triton_launch_mode == 'c_api'
        disabled = self._config.reload_model_disable
        do_load = (remote and not disabled) or (not remote and not c_api)

        retval = True
        if do_load:
            retval = self._do_load_model_variant(variant_config)
        return retval

    def _do_load_model_variant(self, variant_config):
        """
        Loads a model variant in the client
        """
        self._client.wait_for_server_ready(self._config.client_max_retries)

        variant_name = variant_config.get_field('name')
        if self._client.load_model(model_name=variant_name) == -1:
            return False

        if self._client.wait_for_model_ready(
                model_name=variant_name,
                num_retries=self._config.client_max_retries) == -1:
            return False
        return True

    def _get_measurement_if_config_duplicate(self, run_config):
        """
        Checks whether this run config has measurements
        in the state manager's results object
        """

        model_name = run_config.model_name()
        # FIXME: MM-PHASE-1: Assuming all models are identical, so using first model's name
        model_config_name = run_config.model_configs()[0].get_field('name')
        perf_config_str = run_config.perf_config().representation()

        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        # check whether perf config string is a key in result dict
        if model_name not in results:
            return False
        if not self._is_config_in_results(
                # FIXME: MM-PHASE-1: Assuming all models are identical, so using first model's name
                run_config.model_configs()[0]._model_config,
                results[model_name]):
            return False
        measurements = results[model_name][model_config_name][1]

        # For backward compatibility with keys that still have -u,
        # we will remove -u from all keys, convert to set and check
        # perf_config_str is present
        if perf_config_str in set(
                map(PerfAnalyzerConfig.remove_url_from_cli_string,
                    measurements.keys())):
            return measurements[perf_config_str]
        else:
            return None

    def _is_config_in_results(self, config, model_results):
        """
        Returns true if `config` exists in the checkpoint `model_results`
        """

        for result in model_results.values():
            if config == result[0]._model_config:
                return True
        return False

    def profile_model(self, run_config):
        """
        Runs monitors while running perf_analyzer with a specific set of
        arguments. This will profile model inferencing.

        Parameters
        ----------
        run_config : RunConfig
            run_config object corresponding to the model being profiled.

        Returns
        -------
        (dict of lists, list)
            The gpu specific and non gpu metrics
        """

        # TODO: Need to sort the values for batch size and concurrency
        # for correct measurment of the GPU memory metrics.
        perf_output_writer = None if \
            not self._config.perf_output else FileWriter(self._config.perf_output_path)
        perf_config = run_config.perf_config()
        logger.info(f"Profiling model {perf_config['model-names']}...")

        # FIXME: MM-PHASE-1: Assuming all models are identical, so using first model's config
        cpu_only = run_config.model_configs()[0].cpu_only()
        perf_config = run_config.perf_config()

        # Inform user CPU metric(s) are not being collected under CPU mode
        collect_cpu_metrics_expect = cpu_only or len(self._gpus) == 0
        collect_cpu_metrics_actual = len(self._cpu_metrics) > 0
        if collect_cpu_metrics_expect and not collect_cpu_metrics_actual:
            logger.info(
                "CPU metric(s) are not being collected, while this profiling will run on CPU(s)."
            )
        # Warn user about CPU monitor performance issue
        if collect_cpu_metrics_actual:
            logger.warning("CPU metric(s) are being collected.")
            logger.warning(
                "Collecting CPU metric(s) can affect the latency or throughput numbers reported by perf analyzer."
            )

        # Start monitors and run perf_analyzer
        self._start_monitors(cpu_only=cpu_only)
        perf_analyzer_metrics_or_status = self._get_perf_analyzer_metrics(
            perf_config,
            perf_output_writer,
            perf_analyzer_env=run_config.triton_environment())

        # Failed Status
        if perf_analyzer_metrics_or_status == 1:
            self._stop_monitors(cpu_only=cpu_only)
            self._destroy_monitors(cpu_only=cpu_only)
            return None
        else:
            perf_analyzer_metrics = perf_analyzer_metrics_or_status

        # Get metrics for model inference and combine metrics that do not have GPU UUID
        model_gpu_metrics = {}
        if not cpu_only:
            model_gpu_metrics = self._get_gpu_inference_metrics()
        model_cpu_metrics = self._get_cpu_inference_metrics()

        self._destroy_monitors(cpu_only=cpu_only)

        model_non_gpu_metrics = list(perf_analyzer_metrics.values()) + list(
            model_cpu_metrics.values())

        measurement = None
        if model_gpu_metrics is not None and model_non_gpu_metrics is not None:
            measurement = Measurement(gpu_data=model_gpu_metrics,
                                      non_gpu_data=model_non_gpu_metrics,
                                      perf_config=perf_config)
            self._result_manager.add_measurement(run_config, measurement)
        return measurement

    def _start_monitors(self, cpu_only=False):
        """
        Start any metrics monitors
        """

        if not cpu_only:
            try:
                if self._config.use_local_gpu_monitor:
                    self._gpu_monitor = DCGMMonitor(
                        self._gpus, self._config.monitoring_interval,
                        self._gpu_metrics)
                    self._check_triton_and_model_analyzer_gpus()
                else:
                    self._gpu_monitor = RemoteMonitor(
                        self._config.triton_metrics_url,
                        self._config.monitoring_interval, self._gpu_metrics)
                self._gpu_monitor.start_recording_metrics()
            except TritonModelAnalyzerException:
                self._destroy_monitors()
                raise

        self._cpu_monitor = CPUMonitor(self._server,
                                       self._config.monitoring_interval,
                                       self._cpu_metrics)
        self._cpu_monitor.start_recording_metrics()

    def _stop_monitors(self, cpu_only=False):
        """
        Stop any metrics monitors, when we don't need
        to collect the result
        """

        # Stop DCGM Monitor only if there are GPUs available
        if not cpu_only:
            self._gpu_monitor.stop_recording_metrics()
        self._cpu_monitor.stop_recording_metrics()

    def _destroy_monitors(self, cpu_only=False):
        """
        Destroy the monitors created by start
        """

        if not cpu_only:
            if self._gpu_monitor:
                self._gpu_monitor.destroy()
        if self._cpu_monitor:
            self._cpu_monitor.destroy()
        self._gpu_monitor = None
        self._cpu_monitor = None

    def _get_perf_analyzer_metrics(self,
                                   perf_config,
                                   perf_output_writer=None,
                                   perf_analyzer_env=None):
        """
        Gets the aggregated metrics from the perf_analyzer
        Parameters
        ----------
        perf_config : dict
            The keys are arguments to perf_analyzer The values are their
            values
        perf_output_writer : OutputWriter
            Writer that writes the output from perf_analyzer to the output
            stream/file. If None, the output is not written
        perf_analyzer_env : dict
            a dict of name:value pairs for the environment variables with which
            perf_analyzer should be run.

        Raises
        ------
        TritonModelAnalyzerException
        """

        perf_analyzer = PerfAnalyzer(
            path=self._config.perf_analyzer_path,
            config=perf_config,
            max_retries=self._config.perf_analyzer_max_auto_adjusts,
            timeout=self._config.perf_analyzer_timeout,
            max_cpu_util=self._config.perf_analyzer_cpu_util)

        # IF running with C_API, need to set CUDA_VISIBLE_DEVICES here
        if self._config.triton_launch_mode == 'c_api':
            perf_analyzer_env['CUDA_VISIBLE_DEVICES'] = ','.join(
                [gpu.device_uuid() for gpu in self._gpus])

        status = perf_analyzer.run(self._perf_metrics, env=perf_analyzer_env)

        if perf_output_writer:
            perf_output_writer.write(
                '============== Perf Analyzer Launched ==============\n '
                f'Command: perf_analyzer {perf_config.to_cli_string()} \n\n',
                append=True)
            if perf_analyzer.output():
                perf_output_writer.write(perf_analyzer.output() + '\n',
                                         append=True)

        # PerfAnalyzer run was not succesful
        if status == 1:
            return 1

        perf_records = perf_analyzer.get_records()
        perf_record_aggregator = RecordAggregator()
        perf_record_aggregator.insert_all(perf_records)

        return perf_record_aggregator.aggregate()

    def _get_gpu_inference_metrics(self):
        """
        Stops GPU monitor and aggregates any records
        that are GPU specific
        Returns
        -------
        dict
            keys are gpu ids and values are metric values
            in the order specified in self._gpu_metrics
        """

        # Stop and destroy DCGM monitor
        gpu_records = self._gpu_monitor.stop_recording_metrics()

        # Insert all records into aggregator and get aggregated DCGM records
        gpu_record_aggregator = RecordAggregator()
        gpu_record_aggregator.insert_all(gpu_records)

        records_groupby_gpu = {}
        records_groupby_gpu = gpu_record_aggregator.groupby(
            self._gpu_metrics, lambda record: record.device_uuid())

        gpu_metrics = defaultdict(list)
        for _, metric in records_groupby_gpu.items():
            for gpu_uuid, metric_value in metric.items():
                gpu_metrics[gpu_uuid].append(metric_value)
        return gpu_metrics

    def _get_cpu_inference_metrics(self):
        """
        Stops any monitors that just need the records to be aggregated
        like the CPU mmetrics
        """

        cpu_records = self._cpu_monitor.stop_recording_metrics()

        cpu_record_aggregator = RecordAggregator()
        cpu_record_aggregator.insert_all(cpu_records)
        return cpu_record_aggregator.aggregate()

    def _check_triton_and_model_analyzer_gpus(self):
        """
        Check whether Triton Server and Model Analyzer are using the same GPUs
        Raises
        ------
        TritonModelAnalyzerException
            If they are using different GPUs this exception will be raised.
        """

        if self._config.triton_launch_mode != 'remote' and self._config.triton_launch_mode != 'c_api':
            self._client.wait_for_server_ready(self._config.client_max_retries)

            model_analyzer_gpus = [gpu.device_uuid() for gpu in self._gpus]
            triton_gpus = self._get_triton_metrics_gpus()
            if set(model_analyzer_gpus) != set(triton_gpus):
                raise TritonModelAnalyzerException(
                    "'Triton Server is not using the same GPUs as Model Analyzer: '"
                    f"Model Analyzer GPUs {model_analyzer_gpus}, Triton GPUs {triton_gpus}"
                )

    def _get_triton_metrics_gpus(self):
        """
        Uses prometheus to request a list of GPU UUIDs corresponding to the GPUs
        visible to Triton Inference Server
        Parameters
        ----------
        config : namespace
            The arguments passed into the CLI
        """

        triton_prom_str = str(requests.get(
            self._config.triton_metrics_url).content,
                              encoding='ascii')
        metrics = text_string_to_metric_families(triton_prom_str)

        triton_gpus = []
        for metric in metrics:
            if metric.name == 'nv_gpu_utilization':
                for sample in metric.samples:
                    triton_gpus.append(sample.labels['gpu_uuid'])

        return triton_gpus

    @staticmethod
    def get_metric_types(tags):
        """
        Parameters
        ----------
        tags : list of str
            Human readable names for the
            metrics to monitor. They correspond
            to actual record types.
        Returns
        -------
        List
            of record types being monitored
        """

        return [RecordType.get(tag) for tag in tags]

    @staticmethod
    def is_gpu_metric(tag):
        """
        Returns
        ------
        True if the given tag is a supported gpu metric
        False otherwise
        """
        metric = MetricsManager.get_metric_types([tag])[0]
        return metric in DCGMMonitor.model_analyzer_to_dcgm_field

    @staticmethod
    def is_perf_analyzer_metric(tag):
        """
        Returns
        ------
        True if the given tag is a supported perf_analyzer metric
        False otherwise
        """
        metric = MetricsManager.get_metric_types([tag])[0]
        return metric in PerfAnalyzer.perf_metrics

    @staticmethod
    def is_cpu_metric(tag):
        """
        Returns
        ------
        True if the given tag is a supported cpu metric
        False otherwise
        """

        metric = MetricsManager.get_metric_types([tag])[0]
        return metric in CPUMonitor.cpu_metrics
