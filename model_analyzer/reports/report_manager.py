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

from model_analyzer.constants import TOP_MODELS_REPORT_KEY
from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.result.result_table import ResultTable
from .pdf_report import PDFReport

import os
import logging
from numba import cuda
from collections import defaultdict


class ReportManager:
    """
    Manages the building and export of 
    various types of reports
    """
    def __init__(self, config, statistics):
        """
        Parameters
        ----------
        config : AnalyzerConfig
            The model analyzer's config containing information
            about the kind of reports to generate
        statistics : AnalyzerStatistics
            The statistics object that contains information about
            the model anlyzer's runs.
        """

        self._config = config
        self._statistics = statistics

        self._measurements = defaultdict(list)
        self._constraint_strs = self._build_constraint_strings()
        self._reports_export_directory = os.path.join(config.export_path,
                                                      'reports')
        os.makedirs(self._reports_export_directory, exist_ok=True)

    def add_result(self, report_key, result):
        """
        Adds measurements on which the report manager
        can do complex analyses or with which it can
        build tables and add to reports

        Parameters
        ----------
        report_key: str
            The name of the directory that the corresponding
            summary will end up under
        result: ModelResult
            result to be added to report
        """

        for measurement in result.top_n_measurements(n=1):
            model_config = result.model_config()
            self._measurements[report_key].append((model_config, measurement))

    def export_summary(self, report_key):
        """
        Write a PDF summary to disk

        Parameters
        ----------
        report_key : str
            The key for the report we are exporting (e.g. the model name)
        """

        model_report_dir = os.path.join(self._reports_export_directory,
                                        report_key)
        os.makedirs(model_report_dir, exist_ok=True)
        output_filename = os.path.join(model_report_dir, 'result_summary.pdf')
        summary = self._build_summary_report(report_key=report_key)
        logging.info(f"Exporting Summary Report to {output_filename}...")
        summary.write_report(filename=output_filename)

    def _build_summary_report(self, report_key):
        """
        Builder method for a summary
        report.
        """

        summary = PDFReport()

        total_measurements = self._statistics.total_measurements(report_key)
        total_configurations = self._statistics.total_configurations(
            report_key)
        num_best_configs = min(self._config.num_configs_per_model,
                               total_configurations)
        gpu_names, max_memories = self._get_gpu_stats(report_key=report_key)
        static_batch_sizes = ','.join(
            sorted(
                set([
                    str(measurement[1].perf_config()['batch-size'])
                    for measurement in self._measurements[report_key]
                ])))

        constraint_str = "None"
        if self._constraint_strs:
            if report_key in self._constraint_strs:
                constraint_str = self._constraint_strs[report_key]
            elif report_key == TOP_MODELS_REPORT_KEY:
                constraint_str = self._constraint_strs['default']

        table, summary_sentence = self._build_summary_table(
            report_key=report_key,
            num_measurements=total_measurements,
            gpu_name=gpu_names)

        # Add summary sections
        summary.add_title(title="Result Summary")
        summary.add_subheading(f"Model: {report_key}")
        summary.add_paragraph(f"GPUS: {gpu_names}")
        summary.add_paragraph(f"Total Available GPU Memory: {max_memories}")
        summary.add_paragraph(
            f"Client Request Batch Size: {static_batch_sizes}")
        summary.add_paragraph(
            f"Request Protocol: {self._config.client_protocol.upper()}")
        summary.add_paragraph(f"Constraint targets: {constraint_str}")
        summary.add_paragraph(summary_sentence)
        summary.add_paragraph(
            f"Curves corresponding to the {num_best_configs} best model "
            f"configurations out of a total of {total_configurations} are "
            "shown in the plots.")
        throughput_latency_plot = os.path.join(self._config.export_path,
                                               'plots', report_key,
                                               'throughput_v_latency.png')
        caption1 = f"Throughput vs. Latency curves for {num_best_configs} best configurations."
        memory_latency_plot = os.path.join(self._config.export_path, 'plots',
                                           report_key, 'gpu_mem_v_latency.png')
        caption2 = f"GPU Memory vs. Latency curves for {num_best_configs} best configurations."
        summary.add_images([throughput_latency_plot, memory_latency_plot],
                           [caption1, caption2])
        summary.add_paragraph(
            "The maximum GPU memory consumption for each of the above points is"
            f" shown in the second plot. The GPUs {gpu_names} have"
            f" a total available memory of {max_memories} respectively.")

        summary.add_paragraph(
            "The following table summarizes each configuration at the measurement"
            " that optimizes the desired metrics under the given constraints.")
        summary.add_table(table=table)
        return summary

    def _build_summary_table(self, report_key, num_measurements, gpu_name):
        """
        Creates a result table corresponding
        to the best measurements for a particular
        model
        """

        summary_table = ResultTable(headers=[
            'Model Config Name', 'Max Dynamic Batch Size', 'Instance Count',
            'p99 Latency (ms)', 'Throughput (infer/sec)',
            'Max GPU Memory Usage (MB)', 'Average GPU Utilization (%)'
        ],
                                    title="Report Table")

        best = True
        summary_sentence = ""
        for model_config, measurement in self._measurements[report_key]:
            dynamic_batching_str = model_config.dynamic_batching_string()
            dynamic_batch_phrase = "dynamic batching disabled" \
                if dynamic_batching_str == "Disabled" \
                    else f"max dynamic batch size of {dynamic_batching_str}"
            instance_group_str = model_config.instance_group_string()
            if best:
                model_config_dict = model_config.get_config()
                platform = model_config_dict[
                    'backend'] if 'backend' in model_config_dict else model_config_dict[
                        'platform']
                summary_sentence = (
                    f"In {num_measurements} measurements, "
                    f"{instance_group_str} model instances "
                    f"with {dynamic_batch_phrase} "
                    f"on platform {platform} delivers "
                    f"maximum throughput under the given constraints on GPU {gpu_name}."
                )
                best = False

            row = [
                model_config.get_field('name'), dynamic_batching_str,
                instance_group_str,
                measurement.get_metric('perf_latency').value(),
                measurement.get_metric('perf_throughput').value(),
                measurement.get_metric('gpu_used_memory').value(),
                round(measurement.get_metric('gpu_utilization').value(), 1)
            ]
            summary_table.insert_row_by_index(row)
        return summary_table, summary_sentence

    def _get_gpu_stats(self, report_key):
        """
        Gets names and memory infos
        of GPUs used in best measurements
        """

        gpu_names = []
        max_memories = []
        seen_gpus = set()

        for _, measurement in self._measurements[report_key]:
            for gpu in cuda.gpus:
                if gpu.id in measurement.gpus_used(
                ) and gpu.id not in seen_gpus:
                    seen_gpus.add(gpu.id)
                    gpu_names.append((gpu.name).decode('ascii'))
                    with gpu:
                        mems = cuda.current_context().get_memory_info()
                        # convert bytes to GB
                        max_memories.append(round(mems.total / (2**30), 1))

        return ','.join(gpu_names), ','.join(
            [str(x) + ' GB' for x in max_memories])

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
