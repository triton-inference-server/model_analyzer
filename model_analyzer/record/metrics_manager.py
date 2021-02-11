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
import logging

from .record_aggregator import RecordAggregator
from .metrics_mapper import MetricsMapper
from .measurement import Measurement
from model_analyzer.monitor.dcgm.dcgm_monitor import DCGMMonitor
from model_analyzer.monitor.cpu_monitor import CPUMonitor
from model_analyzer.perf_analyzer.perf_analyzer import PerfAnalyzer
from model_analyzer.result.result_comparator import ResultComparator

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
        """

        self._server = server
        self._gpus = config.gpus
        self._monitoring_interval = config.monitoring_interval
        self._perf_analyzer_path = config.perf_analyzer_path
        self._result_manager = result_manager

        # Separates metrics and objectives into related lists

        self._dcgm_metrics = []
        self._perf_metrics = []
        self._cpu_metrics = []

        monitoring_metrics = MetricsMapper.get_metric_types(tags=metric_tags)
        priorities = MetricsMapper.get_metric_types(tags=config.objectives)

        # Set up constraints
        constraint_tags = list(config.constraints.keys())
        constraint_metrics = MetricsMapper.get_metric_types(
            tags=constraint_tags)
        constraints = {
            constraint_metrics[i]: config.constraints[constraint_tags[i]]
            for i in range(len(constraint_tags))
        }

        self._categorize_metrics(metrics=monitoring_metrics)
        self._configure_result_manager(constraints=constraints,
                                       priorities=priorities)

    def _categorize_metrics(self, metrics):
        """
        Splits up monitoring metrics into various 
        categories, defined in ___init___
        """

        for metric in metrics:
            if metric in DCGMMonitor.model_analyzer_to_dcgm_field:
                self._dcgm_metrics.append(metric)
            elif metric in PerfAnalyzer.perf_metrics:
                self._perf_metrics.append(metric)
            elif metric in CPUMonitor.cpu_metrics:
                self._cpu_metrics.append(metric)

    def _configure_result_manager(self, constraints, priorities):
        """
        Requests the result manager to create tables using
        metric categories and sets the result comparator
        """

        self._result_comparator = ResultComparator(
            gpu_metric_types=self._dcgm_metrics,
            non_gpu_metric_types=self._perf_metrics + self._cpu_metrics,
            metric_priorities=priorities)

        self._result_manager.create_tables(
            gpu_specific_metrics=self._dcgm_metrics,
            non_gpu_specific_metrics=self._perf_metrics + self._cpu_metrics,
            aggregation_tag='Max')

        self._result_manager.set_constraints_and_comparator(
            constraints=constraints, comparator=self._result_comparator)

    def profile_server(self, default_value):
        """
        Runs the DCGM monitor on the triton server without the perf_analyzer

        Parameters
        ----------
        default_value : str
            The value to fill in for columns in the table that don't apply to
            profiling server only

        Raises
        ------
        TritonModelAnalyzerException
        """

        self._start_monitors()
        server_gpu_metrics = self._get_gpu_inference_metrics()
        self._result_manager.add_server_data(data=server_gpu_metrics,
                                             default_value=default_value)

    def profile_model(self, perf_config, perf_output_writer=None):
        """
        Runs monitors while running perf_analyzer with a specific set of
        arguments. This will profile model inferencing.

        Parameters
        ----------
        perf_config : dict
            The keys are arguments to perf_analyzer The values are their
            values
        perf_output_writer : OutputWriter
            Writer that writes the output from perf_analyzer to the output
            stream/file. If None, the output is not written
        """

        # Start monitors and run perf_analyzer
        self._start_monitors()
        perf_analyzer_metrics = self._get_perf_analyzer_metrics(
            perf_config, perf_output_writer)

        # Get metrics for model inference and combine metrics that do not have GPU ID
        model_gpu_metrics = self._get_gpu_inference_metrics()
        model_cpu_metrics = self._get_cpu_inference_metrics()
        model_non_gpu_metric_values = list(
            perf_analyzer_metrics.values()) + list(model_cpu_metrics.values())

        # Construct a measurement
        model_measurement = Measurement(
            gpu_data=model_gpu_metrics,
            non_gpu_data=model_non_gpu_metric_values,
            perf_config=perf_config,
            comparator=self._result_comparator)

        self._result_manager.add_model_data(measurement=model_measurement)

    def _start_monitors(self):
        """
        Start any metrics monitors
        """

        self._dcgm_monitor = DCGMMonitor(self._gpus, self._monitoring_interval,
                                         self._dcgm_metrics)
        self._cpu_monitor = CPUMonitor(self._server, self._monitoring_interval,
                                       self._cpu_metrics)

        self._dcgm_monitor.start_recording_metrics()
        self._cpu_monitor.start_recording_metrics()

    def _destroy_monitors(self):
        """
        Destroy the monitors created by start
        """

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
            perf_analyzer = PerfAnalyzer(path=self._perf_analyzer_path,
                                         config=perf_config)
            perf_analyzer.run(self._perf_metrics)
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
        self._destroy_monitors()

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
        self._destroy_monitors()

        cpu_record_aggregator = RecordAggregator()
        cpu_record_aggregator.insert_all(cpu_records)
        return cpu_record_aggregator.aggregate()
