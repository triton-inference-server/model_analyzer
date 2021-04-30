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
from collections import defaultdict

from .record_aggregator import RecordAggregator
from .record import RecordType
from model_analyzer.device.gpu_device_factory import GPUDeviceFactory
from model_analyzer.monitor.dcgm.dcgm_monitor import DCGMMonitor
from model_analyzer.monitor.cpu_monitor import CPUMonitor
from model_analyzer.perf_analyzer.perf_analyzer import PerfAnalyzer
from model_analyzer.result.measurement import Measurement

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class MetricsManager:
    """
    This class handles the profiling
    categorization of metrics
    """
    def __init__(self, config, metric_tags, server, result_manager):
        """
        Parameters
        ----------
        config : AnalyzerConfig
            The model analyzer's config 
        metric_tags : List of str
            The list of tags corresponding to the metric types to monitor.
        server : TritonServer
            Handle to the instance of Triton being used
        result_manager : ResultManager
            instance that manages the result tables and 
            adding results
        gpus : list
            A list of GPUDevice
        """

        self._server = server
        self._gpus = GPUDeviceFactory.get_analyzer_gpus(config.gpus)
        self._monitoring_interval = config.monitoring_interval
        self._perf_analyzer_path = config.perf_analyzer_path
        self._config = config
        self._is_cpu_only = config.cpu_only
        self._result_manager = result_manager

        self._dcgm_metrics = []
        self._perf_metrics = []
        self._cpu_metrics = []

        self._create_metric_tables(metrics=MetricsManager.get_metric_types(
            tags=metric_tags))

    def _create_metric_tables(self, metrics):
        """
        Splits up monitoring metrics into various
        categories, defined in ___init___ and
        requests result manager to make
        corresponding table
        """

        # Separates metrics and objectives into related lists
        for metric in metrics:
            if metric in DCGMMonitor.model_analyzer_to_dcgm_field:
                self._dcgm_metrics.append(metric)
            elif metric in PerfAnalyzer.perf_metrics:
                self._perf_metrics.append(metric)
            elif metric in CPUMonitor.cpu_metrics:
                self._cpu_metrics.append(metric)

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

        self._start_monitors()
        if not self._is_cpu_only:
            server_gpu_metrics = self._get_gpu_inference_metrics()
            self._result_manager.add_server_data(data=server_gpu_metrics)

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

        perf_config = run_config.perf_config()

        # Start monitors and run perf_analyzer
        self._start_monitors()
        perf_analyzer_metrics_or_status = self._get_perf_analyzer_metrics(
            perf_config, perf_output_writer)

        # Failed Status
        if perf_analyzer_metrics_or_status == 1:
            self._stop_monitors()
            self._destroy_monitors()
            return None
        else:
            perf_analyzer_metrics = perf_analyzer_metrics_or_status

        # Get metrics for model inference and combine metrics that do not have GPU ID
        model_gpu_metrics = self._get_gpu_inference_metrics()
        model_cpu_metrics = self._get_cpu_inference_metrics()
        model_non_gpu_metrics = list(perf_analyzer_metrics.values()) + list(
            model_cpu_metrics.values())

        measurement = None
        if model_gpu_metrics is not None and model_non_gpu_metrics is not None:
            measurement = Measurement(gpu_data=model_gpu_metrics,
                                      non_gpu_data=model_non_gpu_metrics,
                                      perf_config=perf_config)
            self._result_manager.add_measurement(run_config, measurement)
        return measurement

    def _start_monitors(self):
        """
        Start any metrics monitors
        """

        # Start DCGM Monitor only if there are GPUs available
        if not self._is_cpu_only:
            self._dcgm_monitor = DCGMMonitor(self._gpus,
                                             self._monitoring_interval,
                                             self._dcgm_metrics)
            self._dcgm_monitor.start_recording_metrics()
        self._cpu_monitor = CPUMonitor(self._server, self._monitoring_interval,
                                       self._cpu_metrics)

        self._cpu_monitor.start_recording_metrics()

    def _stop_monitors(self):
        """
        Stop any metrics monitors
        """

        # Start DCGM Monitor only if there are GPUs available
        if not self._is_cpu_only:
            self._dcgm_monitor.stop_recording_metrics()
        self._cpu_monitor.stop_recording_metrics()

    def _destroy_monitors(self):
        """
        Destroy the monitors created by start
        """

        if not self._is_cpu_only:
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
                path=self._perf_analyzer_path,
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

        if not self._is_cpu_only:
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

            self._destroy_monitors()
            return gpu_metrics
        else:
            self._destroy_monitors()
            return {}

    def _get_cpu_inference_metrics(self):
        """
        Stops any monitors that just need the records to be aggregated
        like the CPU mmetrics
        """

        cpu_records = self._cpu_monitor.stop_recording_metrics()
        self._destroy_monitors()

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
