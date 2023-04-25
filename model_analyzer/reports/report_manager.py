# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, Tuple, Dict, Any, Union, DefaultDict, TYPE_CHECKING

if TYPE_CHECKING:
    from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from model_analyzer.constants import LOGGER_NAME, TOP_MODELS_REPORT_KEY, GLOBAL_CONSTRAINTS_KEY
from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.plots.plot_manager import PlotManager
from model_analyzer.result.result_table import ResultTable
from model_analyzer.config.generate.base_model_config_generator import BaseModelConfigGenerator
from .report_factory import ReportFactory

from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_report import ConfigCommandReport

from model_analyzer.reports.pdf_report import PDFReport
from model_analyzer.reports.html_report import HTMLReport
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

import os
from collections import defaultdict
import logging

logger = logging.getLogger(LOGGER_NAME)


class ReportManager:
    """
    Manages the building and export of 
    various types of reports
    """

    def __init__(self, mode: str, config: Union[ConfigCommandProfile,
                                                ConfigCommandReport],
                 gpu_info: Dict[str, Dict[str,
                                          Any]], result_manager: ResultManager,
                 constraint_manager: ConstraintManager):
        """
        Parameters
        ----------
        mode: str
            The mode in which Model Analyzer is operating
        config :ConfigCommandProfile or ConfigCommandReport
            The model analyzer's config containing information
            about the kind of reports to generate
        gpu_info: dict
            containing information about the GPUs used
            during profiling
        result_manager : ResultManager
            instance that manages the result tables and 
            adding results
        constraint_manager: ConstraintManager
            instance that manages constraints
        """

        self._mode = mode
        self._config = config
        self._gpu_info = gpu_info
        self._result_manager = result_manager
        self._constraint_manager = constraint_manager

        # Create the plot manager
        self._plot_manager = PlotManager(
            config=self._config,
            result_manager=self._result_manager,
            constraint_manager=self._constraint_manager)

        self._summary_data: DefaultDict[str, List[Tuple[
            RunConfig, RunConfigMeasurement]]] = defaultdict(list)
        self._summaries: Dict[str, Union[PDFReport, HTMLReport]] = {}

        self._detailed_report_data: Dict[str, Tuple[RunConfig,
                                                    RunConfigMeasurement]] = {}
        self._detailed_reports: Dict[str, Union[PDFReport, HTMLReport]] = {}

        self._reports_export_directory = os.path.join(config.export_path,
                                                      'reports')
        os.makedirs(self._reports_export_directory, exist_ok=True)

        self._cpu_metrics_gathered_sticky = None

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
        model_names = self._result_manager._profile_model_names

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

        model_names = self._result_manager._profile_model_names

        for model_name in model_names:
            top_results = self._result_manager.top_n_results(
                model_name=model_name,
                n=self._config.num_configs_per_model,
                include_default=True)

            for result in top_results:
                for measurement in result.top_n_measurements(n=1):
                    self._summary_data[model_name].append(
                        (result.run_config(), measurement))

        if self._config.num_top_model_configs:
            for result in self._result_manager.top_n_results(
                    n=self._config.num_top_model_configs):
                for measurement in result.top_n_measurements(n=1):
                    self._summary_data[TOP_MODELS_REPORT_KEY].append(
                        (result.run_config(), measurement))

    def _add_detailed_report_data(self):
        """
        Adds data specific to the model configs
        for which we want detailed reports
        """

        model_config_names = [
            model.model_config_name()
            for model in self._config.report_model_configs
        ]

        # TODO-TMA-650 - this needs to be updated for multi-model
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
            if plot_config.title().startswith(
                    "RAM") and not self._cpu_metrics_were_gathered():
                continue
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

        sort_by_tag = 'latency' if self._mode == 'online' else 'throughput'
        detailed_report.add_paragraph(
            "The first plot above shows the breakdown of the latencies in "
            "the latency throughput curve for this model config. Following that "
            "are the requested configurable plots showing the relationship between "
            "various metrics measured by the Model Analyzer. The above table contains "
            "detailed data for each of the measurements taken for this model config in "
            f"decreasing order of {sort_by_tag}.",
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
        run_config = self._summary_data[report_key][0][0]
        cpu_only = run_config.cpu_only()

        (gpu_names, max_memories) = self._get_gpu_stats(
            measurements=[v for _, v in self._summary_data[report_key]])

        # Get constraints
        constraint_str = self._create_constraint_string(report_key)

        # Build summary table and info sentence
        if not cpu_only:
            table, summary_sentence = self._build_summary_table(
                report_key=report_key,
                num_configurations=total_configurations,
                num_measurements=total_measurements,
                gpu_name=gpu_names)
        else:
            table, summary_sentence = self._build_summary_table(
                report_key=report_key,
                num_configurations=total_configurations,
                num_measurements=total_measurements,
                cpu_only=True)

        # Add summary sections
        summary.add_title(title=f"{self._mode.title()} Result Summary")
        summary.add_subheading(f"Model: {' and '.join(report_key.split(','))}")
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
            if self._mode == 'online' and self._cpu_metrics_were_gathered():
                memory_latency_plot = os.path.join(self._config.export_path,
                                                   'plots', 'simple',
                                                   report_key,
                                                   'cpu_mem_v_latency.png')
                caption_memory_latency = f"CPU Memory vs. Latency curves for {num_best_configs} best configurations."
                summary.add_images([memory_latency_plot],
                                   [caption_memory_latency],
                                   image_width=66)

        caption_results_table = (
            "<div style = \"display:block; clear:both; page-break-after:always;\"></div>"
            "The following table summarizes each configuration at the measurement"
            " that optimizes the desired metrics under the given constraints.")

        if self._result_manager._profiling_models_concurrently():
            caption_results_table = caption_results_table + " Per model values are parenthetical."

        if run_config.is_ensemble_model():
            caption_results_table = caption_results_table + " The ensemble's composing model values are listed in the following order: "
        elif run_config.is_bls_model():
            caption_results_table = caption_results_table + " The BLS composing model values are listed in the following order: "

        if run_config.is_ensemble_model() or run_config.is_bls_model():
            for composing_config_name in run_config.model_run_configs(
            )[0].get_composing_config_names():
                caption_results_table = caption_results_table + BaseModelConfigGenerator.extract_model_name_from_variant_name(
                    composing_config_name) + ", "
            caption_results_table = caption_results_table[:-2]  # removes comma

        summary.add_paragraph(caption_results_table)
        summary.add_table(table=table)

        return summary

    def _build_summary_table(self,
                             report_key,
                             num_configurations,
                             num_measurements,
                             gpu_name=None,
                             cpu_only=False):
        """
        Creates a result table corresponding
        to the best measurements for a particular
        model
        """

        best_run_config, best_run_config_measurement, sorted_measurements = self._find_best_configs(
            report_key)

        multi_model = len(best_run_config.model_run_configs()) > 1
        is_ensemble = best_run_config.is_ensemble_model()
        is_bls = best_run_config.is_bls_model()
        has_composing_models = is_ensemble or is_bls

        summary_sentence = self._create_summary_sentence(
            report_key, num_configurations, num_measurements, best_run_config,
            best_run_config_measurement, gpu_name, cpu_only, multi_model,
            is_ensemble, is_bls)

        summary_table = self._construct_summary_result_table_cpu_only(sorted_measurements, multi_model, has_composing_models) if cpu_only else \
                        self._construct_summary_result_table(sorted_measurements, multi_model, has_composing_models)

        return summary_table, summary_sentence

    def _find_best_configs(self, report_key):
        sorted_measurements = sorted(self._summary_data[report_key],
                                     key=lambda x: x[1],
                                     reverse=True)

        best_run_config = sorted_measurements[0][0]
        best_run_config_measurement = sorted_measurements[0][1]

        return best_run_config, best_run_config_measurement, sorted_measurements

    def _create_constraint_string(self, report_key: str) -> str:
        constraint_strs = self._build_constraint_strings()

        constraint_str = "None"
        if constraint_strs:
            if report_key == TOP_MODELS_REPORT_KEY:
                constraint_str = constraint_strs[GLOBAL_CONSTRAINTS_KEY]
            elif ',' in report_key:  # indicates multi-model
                constraint_str = self._create_multi_model_constraint_string(
                    report_key, constraint_strs)
            else:  # single-model
                if report_key in constraint_strs:
                    constraint_str = constraint_strs[report_key]

        return constraint_str

    def _create_multi_model_constraint_string(
            self, report_key: str, constraint_strs: Dict[str, str]) -> str:
        constraint_str = ""
        for model_name in report_key.split(','):
            if model_name in constraint_strs:
                if constraint_str:
                    constraint_str += "<br>"
                    for i in range(len("Constraint targets: ")):
                        constraint_str += "&ensp;"

                constraint_str += "<strong>" + model_name + "</strong>: " + constraint_strs[
                    model_name]

        return constraint_str

    def _create_summary_sentence(self, report_key, num_configurations,
                                 num_measurements, best_run_config,
                                 best_run_config_measurement, gpu_name,
                                 cpu_only, multi_model, is_ensemble, is_bls):
        measurement_phrase = self._create_summary_measurement_phrase(
            num_measurements)
        config_phrase = self._create_summary_config_phrase(
            best_run_config, num_configurations)
        objective_phrase = self._create_summary_objective_phrase(
            report_key, best_run_config_measurement)
        gpu_name_phrase = self._create_summary_gpu_name_phrase(
            gpu_name, cpu_only)

        summary_sentence = (
            f"In {measurement_phrase} across {config_phrase} "
            f"{objective_phrase}, under the given constraints{gpu_name_phrase}."
        )

        if is_ensemble:
            summary_sentence = summary_sentence + self._create_ensemble_summary_sentence(
                best_run_config)
        elif is_bls:
            summary_sentence = summary_sentence + self._create_bls_summary_sentence(
                best_run_config)
        else:
            summary_sentence = summary_sentence + self._create_model_summary_sentence(
                best_run_config)

        summary_sentence = summary_sentence + ' </UL>'
        return summary_sentence

    def _create_ensemble_summary_sentence(self, run_config: RunConfig) -> str:
        summary_sentence = "<BR><BR>"
        best_config_name = run_config.model_run_configs()[0].model_config(
        ).get_field('name')

        summary_sentence = summary_sentence + f"<strong>{best_config_name}</strong> is comprised of the following composing models: <UL> "
        summary_sentence = summary_sentence + self._create_composing_model_summary_sentence(
            run_config)

        return summary_sentence

    def _create_bls_summary_sentence(self, run_config: RunConfig) -> str:
        summary_sentence = self._create_model_summary_sentence(run_config)
        summary_sentence = summary_sentence + f"<BR>Which is comprised of the following composing models: <UL>"
        summary_sentence = summary_sentence + self._create_composing_model_summary_sentence(
            run_config)

        return summary_sentence

    def _create_model_summary_sentence(self, run_config: RunConfig) -> str:
        summary_sentence = '<UL>'
        for model_run_config in run_config.model_run_configs():
            summary_sentence = summary_sentence + '<LI> ' + self._create_summary_config_info(
                model_run_config.model_config()) + ' </LI>'

        return summary_sentence

    def _create_composing_model_summary_sentence(self,
                                                 run_config: RunConfig) -> str:
        summary_sentence = ""
        for composing_config in run_config.model_run_configs(
        )[0].composing_configs():
            summary_sentence = summary_sentence + '<LI> ' + self._create_summary_config_info(
                composing_config) + ' </LI>'

        return summary_sentence

    def _create_summary_measurement_phrase(self, num_measurements):
        assert num_measurements > 0, "Number of measurements must be greater than 0"

        return f"{num_measurements} measurements" if num_measurements > 1 else \
                "1 measurement"

    def _create_summary_config_phrase(self, best_run_config,
                                      num_configurations):
        config_names = [
            f"<strong>{model_run_config.model_config().get_field('name')}</strong>"
            for model_run_config in best_run_config.model_run_configs()
        ]

        config_names_str = f"{' and '.join(config_names)}"

        if len(config_names) > 1:
            return f"{num_configurations} configurations, the combination of {config_names_str}"
        else:
            return f"{num_configurations} configurations, {config_names_str}"

    def _create_summary_objective_phrase(
            self, report_key: str,
            best_run_config_measurement: "RunConfigMeasurement") -> str:
        default_run_config_measurement = self._find_default_run_config_measurement(
            report_key)

        if default_run_config_measurement:
            objective_gain = self._get_objective_gain(
                best_run_config_measurement, default_run_config_measurement)
        else:
            objective_gain = 0

        if (objective_gain > 0):
            if self._config.get_config()['objectives'].is_set_by_user():
                objective_phrase = f"is <strong>{objective_gain}%</strong> better than the default configuration at meeting the objectives"
            else:
                if self._mode == 'online':
                    objective_phrase = f"is <strong>{objective_gain}%</strong> better than the default configuration at maximizing throughput"
                else:
                    objective_phrase = f"is <strong>{objective_gain}%</strong> better than the default configuration at minimizing latency"
        else:
            objective_phrase = "provides no gain over the default configuration"

        return objective_phrase

    def _get_objective_gain(
            self, run_config_measurement: "RunConfigMeasurement",
            default_run_config_measurement: "RunConfigMeasurement") -> float:
        return round(
            run_config_measurement.calculate_weighted_percentage_gain(
                default_run_config_measurement))

    def _find_default_run_config_measurement(self, model_name):
        # There is no single default config when comparing across
        # multiple model runs
        #
        if model_name == TOP_MODELS_REPORT_KEY:
            return None

        sorted_results = self._result_manager.get_model_sorted_results(
            model_name)

        for run_config_result in sorted_results.results():
            run_config_measurements = run_config_result.passing_measurements()
            if run_config_measurements and 'default' in run_config_measurements[
                    0].model_variants_name():
                best_rcm = run_config_measurements[0]
                for run_config_measurement in run_config_measurements:
                    if run_config_measurement > best_rcm:
                        best_rcm = run_config_measurement

                return best_rcm

        return None

    def _create_summary_platform_phrase(self, model_config):
        if model_config.get_field('backend'):
            platform = model_config.get_field('backend')
        else:
            platform = model_config.get_field('platform')

        return f"platform {platform}"

    def _create_summary_max_batch_size_phrase(self, model_config):
        return f"max batch size of {model_config.max_batch_size()}"

    def _create_instance_group_phrase(self, model_config):
        instance_group_str = model_config.instance_group_string(
            self._get_gpu_count())
        kind_counts = instance_group_str.split('+')
        ret_str = ""
        for kind_count in kind_counts:
            kind_count = kind_count.strip()
            count, kind = kind_count.split(':')
            if ret_str != "":
                ret_str += " and "
            ret_str += f"{count} {kind} instance"
            if int(count) > 1:
                ret_str += "s"
        return ret_str

    def _create_summary_gpu_name_phrase(self, gpu_name, cpu_only):
        return f", on GPU(s) {gpu_name}" if not cpu_only else ""

    def _construct_summary_result_table_cpu_only(self, sorted_measurements,
                                                 multi_model,
                                                 has_composing_models):
        summary_table = self._create_summary_result_table_header_cpu_only(
            multi_model)

        for run_config, run_config_measurement in sorted_measurements:
            row = self._create_summary_row_cpu_only(run_config,
                                                    run_config_measurement,
                                                    has_composing_models)
            summary_table.insert_row_by_index(row)

        return summary_table

    def _construct_summary_result_table(self, sorted_measurements, multi_model,
                                        has_composing_models):
        summary_table = self._create_summary_result_table_header(multi_model)

        for run_config, run_config_measurement in sorted_measurements:
            row = self._create_summary_row(run_config, run_config_measurement,
                                           has_composing_models)
            summary_table.insert_row_by_index(row)

        return summary_table

    def _create_summary_result_table_header_cpu_only(self, multi_model):
        if multi_model:
            header_values = [
                'Model Config Name', 'Max Batch Size', 'Dynamic Batching',
                'Total Instance Count', 'Average p99 Latency (ms)',
                'Total Throughput (infer/sec)', 'Max CPU Memory Usage (MB)'
            ]
        else:
            header_values = [
                'Model Config Name', 'Max Batch Size', 'Dynamic Batching',
                'Total Instance Count', 'p99 Latency (ms)',
                'Throughput (infer/sec)', 'Max CPU Memory Usage (MB)'
            ]
        if not self._cpu_metrics_were_gathered():
            header_values.remove('Max CPU Memory Usage (MB)')

        return ResultTable(headers=header_values, title="Report Table")

    def _create_summary_result_table_header(self, multi_model):
        if multi_model:
            header_values = [
                'Model Config Name', 'Max Batch Size', 'Dynamic Batching',
                'Total Instance Count', 'Average p99 Latency (ms)',
                'Total Throughput (infer/sec)', 'Max CPU Memory Usage (MB)',
                'Max GPU Memory Usage (MB)', 'Average GPU Utilization (%)'
            ]
        else:
            header_values = [
                'Model Config Name', 'Max Batch Size', 'Dynamic Batching',
                'Total Instance Count', 'p99 Latency (ms)',
                'Throughput (infer/sec)', 'Max CPU Memory Usage (MB)',
                'Max GPU Memory Usage (MB)', 'Average GPU Utilization (%)'
            ]

        if not self._cpu_metrics_were_gathered():
            header_values.remove('Max CPU Memory Usage (MB)')

        return ResultTable(headers=header_values, title="Report Table")

    def _create_summary_row_cpu_only(self, run_config, run_config_measurement,
                                     has_composing_models):
        model_config_names = ', '.join([
            model_run_config.model_config().get_field('name')
            for model_run_config in run_config.model_run_configs()
        ])

        if has_composing_models:
            dynamic_batching_string = self._create_summary_string([
                model_config.dynamic_batching_string()
                for model_config in run_config.composing_configs()
            ])
        else:
            dynamic_batching_string = self._create_summary_string([
                model_run_config.model_config().dynamic_batching_string()
                for model_run_config in run_config.model_run_configs()
            ])

        if has_composing_models:
            max_batch_sizes = ', '.join([
                str(model_config.max_batch_size())
                for model_config in run_config.composing_configs()
            ])
        else:
            max_batch_sizes = ', '.join([
                str(model_run_config.model_config().max_batch_size())
                for model_run_config in run_config.model_run_configs()
            ])

        if has_composing_models:
            instance_group_strings = ', '.join([
                model_config.instance_group_string(self._get_gpu_count())
                for model_config in run_config.model_run_configs()
                [0].composing_configs()
            ])
        else:
            instance_group_strings = ', '.join([
                model_run_config.model_config().instance_group_string(
                    self._get_gpu_count())
                for model_run_config in run_config.model_run_configs()
            ])

        perf_latency_string = self._create_non_gpu_metric_string(
            run_config_measurement=run_config_measurement,
            non_gpu_metric='perf_latency_p99')
        perf_throughput_string = self._create_non_gpu_metric_string(
            run_config_measurement=run_config_measurement,
            non_gpu_metric='perf_throughput')

        row = [
            model_config_names, max_batch_sizes, dynamic_batching_string,
            instance_group_strings, perf_latency_string, perf_throughput_string
        ]

        if self._cpu_metrics_were_gathered():
            cpu_used_ram_string = self._create_non_gpu_metric_string(
                run_config_measurement=run_config_measurement,
                non_gpu_metric='cpu_used_ram')
            row.append(cpu_used_ram_string)

        return row

    def _create_summary_row(self, run_config, run_config_measurement,
                            has_composing_models):
        if has_composing_models:
            dynamic_batching_string = self._create_summary_string([
                model_config.dynamic_batching_string()
                for model_config in run_config.composing_configs()
            ])
        else:
            dynamic_batching_string = self._create_summary_string([
                model_run_config.model_config().dynamic_batching_string()
                for model_run_config in run_config.model_run_configs()
            ])

        if has_composing_models:
            instance_group_string = self._create_summary_string([
                model_config.instance_group_string(self._get_gpu_count())
                for model_config in run_config.model_run_configs()
                [0].composing_configs()
            ])
        else:
            instance_group_string = self._create_summary_string([
                model_run_config.model_config().instance_group_string(
                    self._get_gpu_count())
                for model_run_config in run_config.model_run_configs()
            ])

        if has_composing_models:
            max_batch_sizes_string = self._create_summary_string([
                str(model_config.max_batch_size())
                for model_config in run_config.composing_configs()
            ])
        else:
            max_batch_sizes_string = self._create_summary_string([
                str(model_run_config.model_config().max_batch_size())
                for model_run_config in run_config.model_run_configs()
            ])

        model_config_names = '<br>'.join([
            model_run_config.model_config().get_field('name')
            for model_run_config in run_config.model_run_configs()
        ])

        perf_latency_string = self._create_non_gpu_metric_string(
            run_config_measurement=run_config_measurement,
            non_gpu_metric='perf_latency_p99')
        perf_throughput_string = self._create_non_gpu_metric_string(
            run_config_measurement=run_config_measurement,
            non_gpu_metric='perf_throughput')

        if self._cpu_metrics_were_gathered():
            cpu_used_ram_string = self._create_non_gpu_metric_string(
                run_config_measurement=run_config_measurement,
                non_gpu_metric='cpu_used_ram')

            row = [
                model_config_names, max_batch_sizes_string,
                dynamic_batching_string, instance_group_string,
                perf_latency_string, perf_throughput_string,
                cpu_used_ram_string,
                int(
                    run_config_measurement.get_gpu_metric_value(
                        'gpu_used_memory')),
                round(
                    run_config_measurement.get_gpu_metric_value(
                        'gpu_utilization'), 1)
            ]
        else:
            row = [
                model_config_names, max_batch_sizes_string,
                dynamic_batching_string, instance_group_string,
                perf_latency_string, perf_throughput_string,
                int(
                    run_config_measurement.get_gpu_metric_value(
                        'gpu_used_memory')),
                round(
                    run_config_measurement.get_gpu_metric_value(
                        'gpu_utilization'), 1)
            ]

        return row

    def _create_summary_string(self, values):
        if len(values) > 1:
            return f"({', '.join(values)})"
        else:
            return f"{values[0]}"

    def _create_non_gpu_metric_string(self, run_config_measurement,
                                      non_gpu_metric):
        non_gpu_metrics = run_config_measurement.get_non_gpu_metric(
            non_gpu_metric)

        if non_gpu_metrics[0] is None:
            return "0"
        elif len(non_gpu_metrics) > 1:
            non_gpu_metric_config_string = ', '.join([
                str(round(non_gpu_metric.value(), 1))
                for non_gpu_metric in non_gpu_metrics
            ])

            return (
                f"<strong>{round(run_config_measurement.get_non_gpu_metric_value(non_gpu_metric), 1)}</strong> "
                f"({non_gpu_metric_config_string})")
        else:
            return f"{non_gpu_metrics[0].value()}"

    def _create_summary_config_info(self, model_config):
        config_info = f"<strong>{model_config.get_field('name')}</strong>: "
        config_info = config_info + f"{self._create_instance_group_phrase(model_config)} with a "
        config_info = config_info + f"{self._create_summary_max_batch_size_phrase(model_config)} on "
        config_info = config_info + f"{self._create_summary_platform_phrase(model_config)}"

        return config_info

    def _build_detailed_table(self, model_config_name):
        """
        Build the table used in the detailed report
        """

        model_config, measurements = self._detailed_report_data[
            model_config_name]
        sort_by_tag = 'perf_latency_p99' if self._mode == 'online' else 'perf_throughput'
        measurements = sorted(
            measurements,
            key=lambda x: x.get_non_gpu_metric_value(sort_by_tag),
            reverse=True)
        cpu_only = model_config.cpu_only()

        if self._was_measured_with_request_rate(measurements[0]):
            first_column_header = 'Request Rate' if self._mode == 'online' else 'Client Batch Size'
            first_column_tag = 'request-rate-range' if self._mode == 'online' else 'batch-size'
        else:
            first_column_header = 'Request Concurrency' if self._mode == 'online' else 'Client Batch Size'
            first_column_tag = 'concurrency-range' if self._mode == 'online' else 'batch-size'

        if not cpu_only:
            headers = [
                first_column_header, 'p99 Latency (ms)',
                'Client Response Wait (ms)', 'Server Queue (ms)',
                'Server Compute Input (ms)', 'Server Compute Infer (ms)',
                'Throughput (infer/sec)', 'Max CPU Memory Usage (MB)',
                'Max GPU Memory Usage (MB)', 'Average GPU Utilization (%)'
            ]
        else:
            headers = [
                first_column_header, 'p99 Latency (ms)',
                'Client Response Wait (ms)', 'Server Queue (ms)',
                'Server Compute Input (ms)', 'Server Compute Infer (ms)',
                'Throughput (infer/sec)', 'Max CPU Memory Usage (MB)'
            ]

        if not self._cpu_metrics_were_gathered():
            headers.remove('Max CPU Memory Usage (MB)')

        detailed_table = ResultTable(headers, title="Detailed Table")

        # Construct table
        if not cpu_only:
            for measurement in measurements:
                row = [
                    # TODO-TMA-568: This needs to be updated because there will be multiple model configs
                    measurement.model_specific_pa_params()[0][first_column_tag],
                    measurement.get_non_gpu_metric_value('perf_latency_p99'),
                    measurement.get_non_gpu_metric_value(
                        'perf_client_response_wait'),
                    measurement.get_non_gpu_metric_value('perf_server_queue'),
                    measurement.get_non_gpu_metric_value(
                        'perf_server_compute_input'),
                    measurement.get_non_gpu_metric_value(
                        'perf_server_compute_infer'),
                    measurement.get_non_gpu_metric_value('perf_throughput')
                ]
                if self._cpu_metrics_were_gathered():
                    row.append(
                        measurement.get_non_gpu_metric_value('cpu_used_ram'))

                row.append(measurement.get_gpu_metric_value('gpu_used_memory'))
                row.append(
                    round(measurement.get_gpu_metric_value('gpu_utilization'),
                          1))

                detailed_table.insert_row_by_index(row)
        else:
            for measurement in measurements:
                row = [
                    # TODO-TMA-568: This needs to be updated because there will be multiple model configs
                    measurement.model_specific_pa_params()[0][first_column_tag],
                    measurement.get_non_gpu_metric_value('perf_latency_p99'),
                    measurement.get_non_gpu_metric_value(
                        'perf_client_response_wait'),
                    measurement.get_non_gpu_metric_value('perf_server_queue'),
                    measurement.get_non_gpu_metric_value(
                        'perf_server_compute_input'),
                    measurement.get_non_gpu_metric_value(
                        'perf_server_compute_infer'),
                    measurement.get_non_gpu_metric_value('perf_throughput')
                ]
                if self._cpu_metrics_were_gathered():
                    row.append(
                        measurement.get_non_gpu_metric_value('cpu_used_ram'))

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
        instance_group_string = self._create_instance_group_phrase(model_config)
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
            gpu_names, max_memories = self._get_gpu_stats(measurements)
            gpu_cpu_string = f"GPU(s) {gpu_names} with total memory {max_memories}"

        if run_config.is_ensemble_model():
            sentence = f"<strong>{model_config_name}</strong> is comprised of the following composing models:"

            for composing_config in run_config.composing_configs():
                sentence = sentence + '<LI> ' + self._create_summary_config_info(
                    composing_config) + ' </LI>'

            sentence = sentence + f"<br>{len(measurements)} measurement(s) were obtained for the model config on {gpu_cpu_string}."
        elif run_config.is_bls_model():
            sentence = f"<strong>{model_config_name}</strong> is comprised of the following composing models:"

            for composing_config in run_config.composing_configs():
                sentence = sentence + '<LI> ' + self._create_summary_config_info(
                    composing_config) + ' </LI>'

            sentence = sentence + f"<br>{len(measurements)} measurement(s) were obtained for the model config on {gpu_cpu_string}."
        else:
            sentence = (
                f"The model config <strong>{model_config_name}</strong> uses {instance_group_string} "
                f"with {max_batch_size_string} and has {dynamic_batching_string}. "
                f"{len(measurements)} measurement(s) were obtained for the model config on "
                f"{gpu_cpu_string}. "
                f"This model uses the platform {platform}.")

        return sentence

    def _get_gpu_count(self):
        return len(self._gpu_info)

    def _get_gpu_stats(
            self,
            measurements: List["RunConfigMeasurement"]) -> Tuple[str, str]:
        """
        Gets names and max total memory of GPUs used in measurements as a 
        tuple of strings

        Returns
        -------
        (gpu_names_str, max_memory_str):
            The GPU names as a string, and the total combined memory as a string
        """

        gpu_dict: Dict[str, Any] = {}
        for gpu_uuid, gpu_info in self._gpu_info.items():
            for measurement in measurements:
                if gpu_uuid in measurement.gpus_used():
                    gpu_name = gpu_info['name']
                    max_memory = round(gpu_info['total_memory'] / (2**30), 1)
                    if gpu_name not in gpu_dict:
                        gpu_dict[gpu_name] = {"memory": max_memory, "count": 1}
                    else:
                        gpu_dict[gpu_name]["count"] += 1
                    break

        gpu_names = ""
        max_memory = 0
        for name in gpu_dict.keys():
            count = gpu_dict[name]["count"]
            memory = gpu_dict[name]["memory"]
            if gpu_names != "":
                gpu_names += ", "
            gpu_names += f"{count} x {name}"
            max_memory += memory * count

        max_mem_str = f"{max_memory} GB"
        return (gpu_names, max_mem_str)

    def _build_constraint_strings(self) -> Dict[str, str]:
        """
        Constructs constraint strings to show the constraints under which
        each model is being run.
        """

        constraint_strs = {}

        for model_name, model_constraints in self._constraint_manager.get_constraints_for_all_models(
        ).items():
            strs = []
            if model_constraints:
                for metric, constraint in model_constraints.items():
                    metric_header = MetricsManager.get_metric_types(
                        [metric])[0].header(aggregation_tag='')
                    for constraint_type, constraint_val in constraint.items():
                        # String looks like 'Max p99 Latency: 99 ms'
                        metric_header_name = metric_header.rsplit(' ', 1)[0]
                        metric_unit = metric_header.rsplit(' ', 1)[1][1:-1]
                        strs.append(
                            f"{constraint_type.capitalize()} {metric_header_name}: {constraint_val} {metric_unit}"
                        )
                constraint_strs[model_name] = ', '.join(strs)
        return constraint_strs

    def _cpu_metrics_were_gathered(self):
        if self._cpu_metrics_gathered_sticky is None:
            used_ram = None
            if self._detailed_report_data:
                key = list(self._detailed_report_data.keys())[0]
                _, measurements = self._detailed_report_data[key]
                used_ram = measurements[0].get_non_gpu_metric_value(
                    'cpu_used_ram')
            else:
                key = list(self._summary_data.keys())[0]
                _, measurement = self._summary_data[key][0]
                used_ram = measurement.get_non_gpu_metric_value('cpu_used_ram')

            self._cpu_metrics_gathered_sticky = used_ram != 0

        return self._cpu_metrics_gathered_sticky

    def _was_measured_with_request_rate(
            self, measurement: RunConfigMeasurement) -> bool:
        if 'request-rate-range' in measurement.model_specific_pa_params(
        )[0] and measurement.model_specific_pa_params(
        )[0]['request-rate-range']:
            return True
        else:
            return False
