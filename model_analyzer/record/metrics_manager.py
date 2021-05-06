# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from model_analyzer.device.gpu_device_factory import GPUDeviceFactory
from model_analyzer.monitor.dcgm.dcgm_monitor import DCGMMonitor
from model_analyzer.monitor.cpu_monitor import CPUMonitor
from model_analyzer.perf_analyzer.perf_analyzer import PerfAnalyzer
from model_analyzer.result.measurement import Measurement

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from collections import defaultdict
import numba


class MetricsManager:
    """
    This class handles the profiling
    categorization of metrics
    """

    metric_tags = [
        "perf_throughput", "perf_latency", "perf_client_response_wait",
        "perf_client_send_recv", "perf_server_queue",
        "perf_server_compute_input", "perf_server_compute_infer",
        "perf_server_compute_output", "gpu_used_memory", "gpu_free_memory",
        "gpu_utilization", "cpu_used_ram", "cpu_available_ram",
        "gpu_power_usage"
    ]

    def __init__(self, config, server, result_manager):
        """
        Parameters
        ----------
        config :ConfigCommandProfile
            The model analyzer's config 
        server : TritonServer
            Handle to the instance of Triton being used
        result_manager : ResultManager
            instance that manages the result tables and 
            adding results
        """

        self._config = config
        self._server = server
        self._result_manager = result_manager

        self._dcgm_metrics = []
        self._perf_metrics = []
        self._cpu_metrics = []

        # Separates metrics and objectives into related lists
        for metric in MetricsManager.get_metric_types(tags=self.metric_tags):
            if metric in DCGMMonitor.model_analyzer_to_dcgm_field:
                self._dcgm_metrics.append(metric)
            elif metric in PerfAnalyzer.perf_metrics:
                self._perf_metrics.append(metric)
            elif metric in CPUMonitor.cpu_metrics:
                self._cpu_metrics.append(metric)

    def create_metric_tables(self):
        """
        Splits up monitoring metrics into various
        categories, defined in ___init___ and
        requests result manager to make
        corresponding table
        """

        self._result_manager.create_tables(
            gpu_specific_metrics=self._dcgm_metrics,
            non_gpu_specific_metrics=self._perf_metrics + self._cpu_metrics)

    def profile_server(self):
        """
        Runs the DCGM monitor on the triton server without the perf_analyzer

        Raises
        ------
        TritonModelAnalyzerException
        """

        cpu_only = (not numba.cuda.is_available())
        self._start_monitors(cpu_only=cpu_only)
        if not cpu_only:
            server_gpu_metrics = self._get_gpu_inference_metrics()
            self._result_manager.add_server_data(data=server_gpu_metrics)
        self._destroy_monitors(cpu_only=cpu_only)

    def profile_model(self, run_config, perf_output_writer=None):
        """
        Runs monitors while running perf_analyzer with a specific set of
        arguments. This will profile model inferencing.

        Parameters
        ----------
        run_config : RunConfig
            run_config object corresponding to the model being profiled.
        perf_output_writer : OutputWriter
            Writer that writes the output from perf_analyzer to the output
            stream/file. If None, the output is not written

        Returns
        -------
        (dict of lists, list)
            The gpu specific and non gpu metrics
        """

        cpu_only = run_config.model_config().cpu_only()
        perf_config = run_config.perf_config()

        # Start monitors and run perf_analyzer
        self._start_monitors(cpu_only=cpu_only)
        perf_analyzer_metrics_or_status = self._get_perf_analyzer_metrics(
            perf_config, perf_output_writer)

        # Failed Status
        if perf_analyzer_metrics_or_status == 1:
            self._stop_monitors(cpu_only=cpu_only)
            self._destroy_monitors(cpu_only=cpu_only)
            return None
        else:
            perf_analyzer_metrics = perf_analyzer_metrics_or_status

        # Get metrics for model inference and combine metrics that do not have GPU ID
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
            self._dcgm_monitor = DCGMMonitor(
                GPUDeviceFactory.get_analyzer_gpus(self._config.gpus),
                self._config.monitoring_interval, self._dcgm_metrics)
            self._dcgm_monitor.start_recording_metrics()

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
            self._dcgm_monitor.stop_recording_metrics()
        self._cpu_monitor.stop_recording_metrics()

    def _destroy_monitors(self, cpu_only=False):
        """
        Destroy the monitors created by start
        """

        if not cpu_only:
            self._dcgm_monitor.destroy()
        self._cpu_monitor.destroy()

    def _get_perf_analyzer_metrics(self, perf_config, perf_output_writer=None):
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

        Raises
        ------
        TritonModelAnalyzerException
        """

        try:
            perf_analyzer = PerfAnalyzer(
                path=self._config.perf_analyzer_path,
                config=perf_config,
                timeout=self._config.perf_analyzer_timeout,
                max_cpu_util=self._config.perf_analyzer_cpu_util)
            status = perf_analyzer.run(self._perf_metrics)
            # PerfAnalzyer run was not succesful
            if status == 1:
                return 1
        except FileNotFoundError as e:
            raise TritonModelAnalyzerException(
                f"perf_analyzer binary not found : {e}")

        if perf_output_writer:
            perf_output_writer.write(perf_analyzer.output() + '\n')

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
            in the order specified in self._dcgm_metrics
        """

        # Stop and destroy DCGM monitor
        dcgm_records = self._dcgm_monitor.stop_recording_metrics()

        # Insert all records into aggregator and get aggregated DCGM records
        dcgm_record_aggregator = RecordAggregator()
        dcgm_record_aggregator.insert_all(dcgm_records)

        records_groupby_gpu = {}
        records_groupby_gpu = dcgm_record_aggregator.groupby(
            self._dcgm_metrics, lambda record: record.device().device_id())

        gpu_metrics = defaultdict(list)
        for _, metric in records_groupby_gpu.items():
            for gpu_id, metric_value in metric.items():
                gpu_metrics[gpu_id].append(metric_value)

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
        metric = MetricsManager.get_metric_types([tag])
        return metric in DCGMMonitor.model_analyzer_to_dcgm_field

    @staticmethod
    def is_perf_analyzer_metric(tag):
        """
        Returns
        ------
        True if the given tag is a supported perf_analyzer metric
        False otherwise
        """
        metric = MetricsManager.get_metric_types([tag])
        return metric in PerfAnalyzer.perf_metrics

    @staticmethod
    def is_perf_analyzer_metric(tag):
        """
        Returns
        ------
        True if the given tag is a supported cpu metric
        False otherwise
        """

        metric = MetricsManager.get_metric_types([tag])
        return metric in CPUMonitor.cpu_metrics
