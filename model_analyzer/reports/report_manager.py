# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.constants import LOGGER_NAME, TOP_MODELS_REPORT_KEY
from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.plots.plot_manager import PlotManager
from model_analyzer.result.result_table import ResultTable
from .report_factory import ReportFactory

import os
from collections import defaultdict
import logging

logger = logging.getLogger(LOGGER_NAME)


class ReportManager:
    """
    Manages the building and export of 
    various types of reports
    """

    def __init__(self, mode, config, gpu_info, result_manager):
        """
        Parameters
        ----------
        mode: str
            The mode in which Model Analyzer is operating
        config :ConfigCommandProfile
            The model analyzer's config containing information
            about the kind of reports to generate
        gpu_info: dict
            containing information about the GPUs used
            during profiling
        result_manager : ResultManager
            instance that manages the result tables and 
            adding results
        """

        self._mode = mode
        self._config = config
        self._gpu_info = gpu_info
        self._result_manager = result_manager

        # Create the plot manager
        self._plot_manager = PlotManager(config=self._config,
                                         result_manager=self._result_manager)

        self._summary_data = defaultdict(list)
        self._summaries = {}

        self._detailed_report_data = {}
        self._detailed_reports = {}

        self._reports_export_directory = os.path.join(config.export_path,
                                                      'reports')
        os.makedirs(self._reports_export_directory, exist_ok=True)

    def report_keys(self):
        """
        Returns
        -------
        list of str
            identifiers for all the
            reports in this report manager
        """

        return list(self._summary_data.keys())

    def data(self, report_key):
        """
        Parameters
        ----------
        report_key: str
            An identifier for a particular report
        Returns
        -------
        dict
            The data in the report corresponding with
            the report key
        """

        return self._summary_data[report_key]

    def create_summaries(self):
        """
        Add summary data and build summary report
        """

        self._add_summary_data()
        self._plot_manager.create_summary_plots()
        self._plot_manager.export_summary_plots()

        statistics = self._result_manager.get_result_statistics()
        model_names = [
            model.model_name() for model in self._config.analysis_models
        ]

        at_least_one_summary = False
        for model_name in model_names:
            if model_name in self._summary_data:
                at_least_one_summary = True
                self._summaries[model_name] = self._build_summary_report(
                    report_key=model_name,
                    num_configs=self._config.num_configs_per_model,
                    statistics=statistics)
            else:
                logger.warning(
                    f'No data found for model {model_name}, skipping export summary.'
                )

        if self._config.num_top_model_configs and at_least_one_summary:
            self._summaries[TOP_MODELS_REPORT_KEY] = self._build_summary_report(
                report_key=TOP_MODELS_REPORT_KEY,
                num_configs=self._config.num_top_model_configs,
                statistics=statistics)

    def export_summaries(self):
        """
        Write a summary to disk
        """

        for report_key, summary in self._summaries.items():
            model_report_dir = os.path.join(self._reports_export_directory,
                                            'summaries', report_key)
            os.makedirs(model_report_dir, exist_ok=True)
            output_filename = os.path.join(
                model_report_dir,
                f'result_summary.{summary.get_file_extension()}')
            logger.info(f"Exporting Summary Report to {output_filename}")
            summary.write_report(filename=output_filename)

    def create_detailed_reports(self):
        """
        Adds detailed report data and build detailed reports 
        """

        self._add_detailed_report_data()
        self._plot_manager.create_detailed_plots()
        self._plot_manager.export_detailed_plots()

        for report_model_config in self._config.report_model_configs:
            model_config_name = report_model_config.model_config_name()
            self._detailed_reports[
                model_config_name] = self._build_detailed_report(
                    report_model_config)

    def export_detailed_reports(self):
        """
        Write a detailed report to disk
        """

        for report_key, report in self._detailed_reports.items():
            model_report_dir = os.path.join(self._reports_export_directory,
                                            'detailed', report_key)
            os.makedirs(model_report_dir, exist_ok=True)
            output_filename = os.path.join(
                model_report_dir,
                f'detailed_report.{report.get_file_extension()}')
            logger.info(f"Exporting Detailed Report to {output_filename}")
            report.write_report(filename=output_filename)

    def _add_summary_data(self):
        """
        Adds measurements on which the report manager
        can do complex analyses or with which it can
        build tables and add to reports
        """

        model_names = [
            model.model_name() for model in self._config.analysis_models
        ]

        for model_name in model_names:
            top_results = self._result_manager.top_n_results(
                model_name=model_name,
                n=self._config.num_configs_per_model,
                include_default=True)

            # TODO-TMA-568: This needs to be updated because there will be multiple model configs
            for result in top_results:
                model_config = result.run_config().model_run_configs(
                )[0].model_config()
                for measurement in result.top_n_measurements(n=1):
                    self._summary_data[model_name].append(
                        (model_config, measurement))

        # TODO-TMA-568: This needs to be updated because there will be multiple model configs
        if self._config.num_top_model_configs:
            for result in self._result_manager.top_n_results(
                    n=self._config.num_top_model_configs):
                model_config = result.run_config().model_run_configs(
                )[0].model_config()
                for measurement in result.top_n_measurements(n=1):
                    self._summary_data[TOP_MODELS_REPORT_KEY].append(
                        (model_config, measurement))

    def _add_detailed_report_data(self):
        """
        Adds data specific to the model configs
        for which we want detailed reports
        """

        model_config_names = [
            model.model_config_name()
            for model in self._config.report_model_configs
        ]

        # TODO-TMA-568 - this likely needs to be updated for multi-model
        for model_config_name in model_config_names:
            self._detailed_report_data[
                model_config_name] = self._result_manager.get_model_configs_run_config_measurements(
                    model_config_name)

    def _build_detailed_report(self, report_model_config):
        """
        Builder method for a detailed report
        """

        detailed_report = ReportFactory.create_report()

        report_key = report_model_config.model_config_name()
        model_config, _ = self._detailed_report_data[report_key]

        detailed_report.add_title(title="Detailed Report")
        detailed_report.add_subheading(subheading=f"Model Config: {report_key}")

        if self._mode == 'online':
            # Add main latency breakdown image
            detailed_plot = os.path.join(self._config.export_path, 'plots',
                                         'detailed', report_key,
                                         'latency_breakdown.png')
            detailed_caption = f"Latency Breakdown for Online Performance of {report_key}"

            # First add row of detailed
            detailed_report.add_images([detailed_plot], [detailed_caption])

        # Next add the SimplePlots created for this detailed report
        plot_stack = []
        caption_stack = []
        plot_path = os.path.join(self._config.export_path, 'plots', 'simple',
                                 report_key)
        for plot_config in report_model_config.plots():
            if model_config.cpu_only() and (
                    plot_config.y_axis().startswith('gpu_') or
                    plot_config.x_axis().startswith('gpu_')):
                continue
            plot_stack.append(
                os.path.join(plot_path, f"{plot_config.name()}.png"))
            caption_stack.append(
                f"{plot_config.title()} curves for config {report_key}")
            if len(plot_stack) == 2:
                detailed_report.add_images(plot_stack,
                                           caption_stack,
                                           float="left")
                plot_stack = []
                caption_stack = []

        # Odd number of plots
        if plot_stack:
            detailed_report.add_images(plot_stack, caption_stack, float="left")

        # Next add table of measurements
        detailed_table = self._build_detailed_table(report_key)
        detailed_report.add_table(table=detailed_table)

        # Add some details about the config
        detailed_info = self._build_detailed_info(report_key)
        detailed_report.add_line_breaks(num_breaks=2)
        detailed_report.add_paragraph(detailed_info, font_size=18)
        detailed_report.add_paragraph(
            "The first plot above shows the breakdown of the latencies in "
            "the latency throughput curve for this model config. Following that "
            "are the requested configurable plots showing the relationship between "
            "various metrics measured by the Model Analyzer. The above table contains "
            "detailed data for each of the measurements taken for this model config in "
            "decreasing order of throughput.",
            font_size=18)
        return detailed_report

    def _build_summary_report(self, report_key, num_configs, statistics):
        """
        Builder method for a summary
        report.
        """

        summary = ReportFactory.create_report()

        total_measurements = statistics.total_measurements(report_key)
        total_configurations = statistics.total_configurations(report_key)
        num_best_configs = min(num_configs, total_configurations)

        # Get GPU names and memory
        model_config = self._summary_data[report_key][0][0]
        cpu_only = model_config.cpu_only()
        gpu_dict = self._get_gpu_stats(
            measurements=[v for _, v in self._summary_data[report_key]])

        gpu_names = ','.join(list(gpu_dict.keys()))
        max_memories = ','.join([str(x) + ' GB' for x in gpu_dict.values()])

        # Get constraints
        constraint_strs = self._build_constraint_strings()
        constraint_str = "None"
        if constraint_strs:
            if report_key in constraint_strs:
                constraint_str = constraint_strs[report_key]
            elif report_key == TOP_MODELS_REPORT_KEY:
                constraint_str = constraint_strs['default']

        # Build summary table and info sentence
        if not cpu_only:
            table, summary_sentence = self._build_summary_table(
                report_key=report_key,
                num_measurements=total_measurements,
                gpu_name=gpu_names)
        else:
            table, summary_sentence = self._build_summary_table(
                report_key=report_key,
                num_measurements=total_measurements,
                cpu_only=True)

        # Add summary sections
        summary.add_title(title=f"{self._mode.title()} Result Summary")
        summary.add_subheading(f"Model: {report_key}")
        if not cpu_only:
            summary.add_paragraph(f"GPU(s): {gpu_names}")
            summary.add_paragraph(f"Total Available GPU Memory: {max_memories}")
        summary.add_paragraph(f"Constraint targets: {constraint_str}")
        summary.add_paragraph(summary_sentence)
        summary.add_paragraph(
            f"Curves corresponding to the {num_best_configs} best model "
            f"configuration(s) out of a total of {total_configurations} are "
            "shown in the plots.")

        throughput_plot_config = self._config.plots[0]
        throughput_plot = os.path.join(self._config.export_path, 'plots',
                                       'simple', report_key,
                                       f'{throughput_plot_config.name()}.png')

        caption_throughput = f"{throughput_plot_config.title()} curves for {num_best_configs} best configurations."

        if not cpu_only:

            summary.add_images([throughput_plot], [caption_throughput],
                               image_width=66)
            if self._mode == 'online':
                memory_latency_plot = os.path.join(self._config.export_path,
                                                   'plots', 'simple',
                                                   report_key,
                                                   'gpu_mem_v_latency.png')
                caption_memory_latency = f"GPU Memory vs. Latency curves for {num_best_configs} best configurations."
                summary.add_images([memory_latency_plot],
                                   [caption_memory_latency],
                                   image_width=66)
        else:
            summary.add_images([throughput_plot], [caption_throughput],
                               image_width=66)
            if self._mode == 'online':
                memory_latency_plot = os.path.join(self._config.export_path,
                                                   'plots', 'simple',
                                                   report_key,
                                                   'cpu_mem_v_latency.png')
                caption_memory_latency = f"CPU Memory vs. Latency curves for {num_best_configs} best configurations."
                summary.add_images([memory_latency_plot],
                                   [caption_memory_latency],
                                   image_width=66)

        summary.add_paragraph(
            "<div style = \"display:block; clear:both; page-break-after:always;\"></div>"
            "The following table summarizes each configuration at the measurement"
            " that optimizes the desired metrics under the given constraints.")
        summary.add_table(table=table)
        return summary

    def _get_dynamic_batching_phrase(self, config):
        dynamic_batching_str = config.dynamic_batching_string()
        assert dynamic_batching_str == "Disabled" or dynamic_batching_str == "Enabled", f"dynamic batching unknown"
        if dynamic_batching_str == "Disabled":
            dynamic_batch_phrase = "dynamic batching disabled"
        else:
            dynamic_batch_phrase = "dynamic batching enabled"
        return dynamic_batch_phrase

    def _build_summary_table(self,
                             report_key,
                             num_measurements,
                             gpu_name=None,
                             cpu_only=False):
        """
        Creates a result table corresponding
        to the best measurements for a particular
        model
        """

        if not cpu_only:
            summary_table = ResultTable(headers=[
                'Model Config Name', 'Max Batch Size', 'Dynamic Batching',
                'Instance Count', 'p99 Latency (ms)', 'Throughput (infer/sec)',
                'Max CPU Memory Usage (MB)', 'Max GPU Memory Usage (MB)',
                'Average GPU Utilization (%)'
            ],
                                        title="Report Table")
        else:
            summary_table = ResultTable(headers=[
                'Model Config Name', 'Max Batch Size', 'Dynamic Batching',
                'Instance Count', 'p99 Latency (ms)', 'Throughput (infer/sec)',
                'Max CPU Memory Usage (MB)'
            ],
                                        title="Report Table")

        sorted_measurements = sorted(self._summary_data[report_key],
                                     key=lambda x: x[1])

        # Construct summary sentence using best config
        best_config = sorted_measurements[0][0]
        model_config_dict = best_config.get_config()
        platform = model_config_dict['backend'] if \
            'backend' in model_config_dict \
            else model_config_dict['platform']
        max_batch_size_phrase = f"max batch size of {best_config.max_batch_size()}"
        dynamic_batch_phrase = self._get_dynamic_batching_phrase(best_config)

        summary_sentence = (
            f"In {num_measurements} measurement(s), "
            f"config {best_config.get_field('name')} ("
            f"{best_config.instance_group_string()} model instance(s) "
            f"with {max_batch_size_phrase} and {dynamic_batch_phrase}) "
            f"on platform {platform} delivers maximum throughput under "
            f"the given constraints{' on GPU(s) '+gpu_name if not best_config.cpu_only() else ''}."
        )

        # Construct table
        if not cpu_only:
            for model_config, measurement in sorted_measurements:
                instance_group_str = model_config.instance_group_string()
                max_batch_size = model_config.max_batch_size()
                row = [
                    model_config.get_field('name'), max_batch_size,
                    model_config.dynamic_batching_string(), instance_group_str,
                    measurement.get_metric_value('perf_latency_p99'),
                    measurement.get_metric_value('perf_throughput'),
                    measurement.get_metric_value('cpu_used_ram'),
                    measurement.get_metric_value('gpu_used_memory'),
                    round(measurement.get_metric_value('gpu_utilization'), 1)
                ]
                summary_table.insert_row_by_index(row)
        else:
            for model_config, measurement in sorted_measurements:
                instance_group_str = model_config.instance_group_string()
                max_batch_size = model_config.max_batch_size()
                row = [
                    model_config.get_field('name'), max_batch_size,
                    model_config.dynamic_batching_string(), instance_group_str,
                    measurement.get_metric_value('perf_latency_p99'),
                    measurement.get_metric_value('perf_throughput'),
                    measurement.get_metric_value('cpu_used_ram')
                ]
                summary_table.insert_row_by_index(row)
        return summary_table, summary_sentence

    def _build_detailed_table(self, model_config_name):
        """
        Build the table used in the detailed report
        """

        model_config, measurements = self._detailed_report_data[
            model_config_name]
        sort_by_tag = 'perf_latency_p99' if self._mode == 'online' else 'perf_throughput'
        measurements = sorted(measurements,
                              key=lambda x: x.get_metric_value(sort_by_tag),
                              reverse=True)
        cpu_only = model_config.cpu_only()

        first_column_header = 'Request Concurrency' if self._mode == 'online' else 'Client Batch Size'
        first_column_tag = 'concurrency-range' if self._mode == 'online' else 'batch-size'
        if not cpu_only:
            detailed_table = ResultTable(headers=[
                first_column_header, 'p99 Latency (ms)',
                'Client Response Wait (ms)', 'Server Queue (ms)',
                'Server Compute Input (ms)', 'Server Compute Infer (ms)',
                'Throughput (infer/sec)', 'Max CPU Memory Usage (MB)',
                'Max GPU Memory Usage (MB)', 'Average GPU Utilization (%)'
            ],
                                         title="Detailed Table")
        else:
            detailed_table = ResultTable(headers=[
                first_column_header, 'p99 Latency (ms)',
                'Client Response Wait (ms)', 'Server Queue (ms)',
                'Server Compute Input (ms)', 'Server Compute Infer (ms)',
                'Throughput (infer/sec)', 'Max CPU Memory Usage (MB)'
            ],
                                         title="Detailed Table")
        # Construct table
        if not cpu_only:
            for measurement in measurements:
                row = [
                    # TODO-TMA-568: This needs to be updated because there will be multiple model configs
                    measurement.model_specific_pa_params()[0][first_column_tag],
                    measurement.get_metric_value('perf_latency_p99'),
                    measurement.get_metric_value('perf_client_response_wait'),
                    measurement.get_metric_value('perf_server_queue'),
                    measurement.get_metric_value('perf_server_compute_input'),
                    measurement.get_metric_value('perf_server_compute_infer'),
                    measurement.get_metric_value('perf_throughput'),
                    measurement.get_metric_value('cpu_used_ram'),
                    measurement.get_metric_value('gpu_used_memory'),
                    round(measurement.get_metric_value('gpu_utilization'), 1)
                ]
                detailed_table.insert_row_by_index(row)
        else:
            for measurement in measurements:
                row = [
                    # TODO-TMA-568: This needs to be updated because there will be multiple model configs
                    measurement.model_specific_pa_params()[0][first_column_tag],
                    measurement.get_metric_value('perf_latency_p99'),
                    measurement.get_metric_value('perf_client_response_wait'),
                    measurement.get_metric_value('perf_server_queue'),
                    measurement.get_metric_value('perf_server_compute_input'),
                    measurement.get_metric_value('perf_server_compute_infer'),
                    measurement.get_metric_value('perf_throughput'),
                    measurement.get_metric_value('cpu_used_ram')
                ]
                detailed_table.insert_row_by_index(row)
        return detailed_table

    def _build_detailed_info(self, model_config_name):
        """
        Constructs important info sentence about the model config
        specified
        """

        run_config, measurements = self._detailed_report_data[model_config_name]

        # TODO-TMA-568 - add support for multi-model
        model_config = run_config.model_run_configs()[0].model_config()
        instance_group_string = model_config.instance_group_string()
        dynamic_batching = model_config.dynamic_batching_string()
        max_batch_size = model_config.max_batch_size()
        platform = model_config.get_field('platform')

        max_batch_size_string = f"a max batch size of {max_batch_size}"

        if dynamic_batching == 'Disabled':
            dynamic_batching_string = "dynamic batching disabled"
        else:
            dynamic_batching_string = "dynamic batching enabled"

        gpu_cpu_string = "CPU"

        if not run_config.cpu_only():
            gpu_dict = self._get_gpu_stats(measurements=measurements)
            gpu_names = ','.join(list(gpu_dict.keys()))
            max_memories = ','.join([str(x) + ' GB' for x in gpu_dict.values()])
            gpu_cpu_string = f"GPU(s) {gpu_names} with memory limit(s) {max_memories}"
        sentence = (
            f"The model config \"{model_config_name}\" uses {instance_group_string.replace('/', ' ')} "
            f"instance(s) with {max_batch_size_string} and has {dynamic_batching_string}. "
            f"{len(measurements)} measurement(s) were obtained for the model config on "
            f"{gpu_cpu_string}. "
            f"This model uses the platform {platform}.")

        return sentence

    def _get_gpu_stats(self, measurements):
        """
        Gets names and memory infos of GPUs used in measurements
        """

        gpu_dict = {}
        for measurement in measurements:
            for gpu_uuid, gpu_info in self._gpu_info.items():
                if gpu_uuid in measurement.gpus_used():
                    gpu_name = gpu_info['name']
                    max_memory = round(gpu_info['total_memory'] / (2**30), 1)
                    if gpu_name not in gpu_dict:
                        gpu_dict[gpu_name] = max_memory
        return gpu_dict

    def _build_constraint_strings(self):
        """
        Constructs constraint strings to show the constraints under which
        each model is being run.
        """

        constraint_strs = {}
        for model_name, model_constraints in ConstraintManager.get_constraints_for_all_models(
                self._config).items():
            strs = []
            if model_constraints:
                for metric, constraint in model_constraints.items():
                    metric_header = MetricsManager.get_metric_types(
                        [metric])[0].header(aggregation_tag='')
                    for constraint_type, constraint_val in constraint.items():
                        # String looks like 'Max p99 Latency : 99 ms'
                        metric_header_name = metric_header.rsplit(' ', 1)[0]
                        metric_unit = metric_header.rsplit(' ', 1)[1][1:-1]
                        strs.append(
                            f"{constraint_type.capitalize()} {metric_header_name} : {constraint_val} {metric_unit}"
                        )
                constraint_strs[model_name] = ', '.join(strs)
        return constraint_strs
