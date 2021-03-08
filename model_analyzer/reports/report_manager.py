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

    def __init__(self, config):
        """
        Parameters
        ----------
        config : AnalyzerConfig
            The model analyzer's config containing information
            about the kind of reports to generate
        """

        self._measurements = defaultdict(list)
        self._config = config
        self._constraint_strs = {}
        self._reports_export_directory = os.path.join(config.export_path,
                                                      'reports')
        os.makedirs(self._reports_export_directory, exist_ok=True)

    def add_measurement(self, model_name, model_config, measurement):
        """
        Adds measurements on which the report manager
        can do complex analyses or with which it can
        build tables and add to reports

        Parameters
        ----------
        model_name : str
            The name of the running model
        model_config : ModelConfig
            the config of the running model
        measurement : Measurement
            The measurement to be added
        """

        self._measurements[model_name].append((model_config, measurement))

    def export_summary(self,
                       statistics,
                       filename='model_analyzer_results_summary.pdf'):
        """
        Write a PDF summary to disk

        Parameters
        ----------
        statistics: AnalyzerStatistics
            Object containing all necessary
            information about this analyzer run
        filename : str
            The filename for the PDF
        """

        summary = self._build_summary_report(statistics=statistics)
        output_filename = os.path.join(self._reports_export_directory,
                                       filename)
        logging.info(f"Exporting Summary Report to {output_filename}...")
        summary.write_report(filename=output_filename)

    def _build_summary_report(self, statistics):
        """
        Builder method for a summary
        report.
        """

        self._build_constraint_strings()
        total_configurations = statistics.total_configurations()
        passing_configurations = statistics.passing_configurations()
        num_best_configs = self._config.top_n_configs

        summary = PDFReport()
        summary.add_title(title="Result Summary")
        for model_name in self._measurements:
            gpu_names, max_memories = self._get_gpu_stats(
                model_name=model_name)
            static_batch_size = self._measurements[model_name][0][
                1].perf_config()['batch-size']
            summary.add_subheading(f"Model: {model_name}")
            summary.add_paragraph(f"GPUS: {gpu_names}")
            summary.add_paragraph(f"Total GPU Memory: {max_memories}")
            summary.add_paragraph(
                f"Client Request Batch Size: {static_batch_size}")
            summary.add_paragraph(
                f"Request Protocol: {self._config.client_protocol}")
            summary.add_paragraph(
                f"Constraint targets: {self._constraint_strs[model_name]}")
            summary.add_paragraph(
                f"Model Analyzer evaluated {total_configurations} model configurations"
                f" and found {passing_configurations} that satisfy your constraints."
                f" Of these {passing_configurations} configurations, the {num_best_configs}"
                " best model configurations are shown in the plots.")
            throughput_latency_plot = os.path.join(self._config.export_path,
                                                   'plots', model_name,
                                                   'throughput_v_latency.png')
            caption1 = f"Throughput vs. Latency curves for {num_best_configs} configurations of {model_name}"
            memory_latency_plot = os.path.join(self._config.export_path,
                                               'plots', model_name,
                                               'gpu_mem_v_latency.png')
            caption2 = f"GPU Memory vs. Latency curves for {num_best_configs} configurations of {model_name}"
            summary.add_images([throughput_latency_plot, memory_latency_plot],
                               [caption1, caption2])
            summary.add_paragraph(
                "The maximum GPU memory consumption for each of the above points is"
                f" shown in the second plot. The GPUs {gpu_names} have"
                f" a total available memory of {max_memories} respectively.")

            summary.add_paragraph(
                "The following table summarizes each configuration at the measurement"
                " that optimizes the desired metrics under the given constraints in"
                " decreasing order of throughput.")
            summary.add_table(table=self._build_summary_table(
                model_name=model_name))
        return summary

    def _build_summary_table(self, model_name):
        """
        Creates a result table corresponding
        to the best measurements for a particular
        model
        """

        summary_table = ResultTable(headers=[
            'Model Config Name', 'Dynamic Batcher Sizes', 'Instance Count',
            'p99 Latency (ms)', 'Throughput (infer/sec)',
            'GPU Memory Usage (MB)', 'GPU Utilization (%)'
        ],
                                    title="Report Table")

        for model_config, measurement in self._measurements[model_name]:
            dynamic_batching_str = model_config.dynamic_batching_string()
            instance_group_str = model_config.instance_group_string()
            row = [
                model_config.get_field('name'), dynamic_batching_str,
                instance_group_str,
                measurement.get_value_of_metric('perf_latency').value(),
                measurement.get_value_of_metric('perf_throughput').value(),
                measurement.get_value_of_metric('gpu_used_memory').value(),
                measurement.get_value_of_metric('gpu_utilization').value()
            ]
            summary_table.insert_row_by_index(row)
        return summary_table

    def _get_gpu_stats(self, model_name):
        """
        Gets names and memory infos 
        of GPUs used in best measurements
        """

        gpu_names = []
        max_memories = []
        seen_gpus = set()

        for _, measurement in self._measurements[model_name]:
            for gpu in cuda.gpus:
                if gpu.id in measurement.gpus_used(
                ) and gpu.id not in seen_gpus:
                    seen_gpus.add(gpu.id)
                    gpu_names.append((gpu.name).decode('ascii'))
                    with gpu:
                        mems = cuda.current_context().get_memory_info()
                        # convert bytes to GB
                        max_memories.append(mems.total // (2**30))

        return ','.join(gpu_names), ','.join(
            [str(x) + ' GB' for x in max_memories])

    def _build_constraint_strings(self):
        """
        Constructs constraint strings to show the constraints under which
        each model is being run.
        """

        for model_name, model_constraints in ConstraintManager.get_constraints_for_all_models(
                self._config).items():
            constraint_strs = []
            for metric, constraint in model_constraints.items():
                metric_header = MetricsManager.get_metric_types(
                    [metric])[0].header(aggregation_tag='')
                for constraint_type, constraint_val in constraint.items():
                    # String looks like 'Max p99 Latency : 99 ms'
                    metric_header_name = metric_header.rsplit(' ', 1)[0]
                    metric_unit = metric_header.rsplit(' ', 1)[1]
                    constraint_strs.append(
                        f"{constraint_type.capitalize()} {metric_header_name} : {constraint_val} {metric_unit}"
                    )
            self._constraint_strs[model_name] = ', '.join(constraint_strs)
