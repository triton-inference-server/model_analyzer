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

from typing import Union, DefaultDict, Dict

from model_analyzer.config.input.config_defaults import DEFAULT_CPU_MEM_PLOT
from model_analyzer.config.input.objects.config_plot import ConfigPlot
from model_analyzer.constants import TOP_MODELS_REPORT_KEY, GLOBAL_CONSTRAINTS_KEY

from .simple_plot import SimplePlot
from .detailed_plot import DetailedPlot

from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_report import ConfigCommandReport

import os
from collections import defaultdict


class PlotManager:
    """
    This class manages the construction and arrangement
    of plots generated by model analyzer
    """

    def __init__(self, config:  Union[ConfigCommandProfile, ConfigCommandReport],
                 result_manager: ResultManager,
                 constraint_manager: ConstraintManager):
        """
        Parameters
        ----------
        config : ConfigCommandProfile or ConfigCommandReport
            The model analyzer's config containing information
            about the kind of plots to generate
        result_manager : ResultManager
            instance that manages the result tables and
            adding results
        constraint_manager: ConstraintManager
            instance that manages constraints
        """

        self._config = config
        self._result_manager = result_manager

        # Constraints should be plotted as well
        self._constraints = constraint_manager.get_constraints_for_all_models()

        # Construct plot output directory
        self._plot_export_directory = os.path.join(config.export_path, 'plots')
        os.makedirs(self._plot_export_directory, exist_ok=True)

        # Dict of list of plots
        self._simple_plots: DefaultDict[str, Dict[str, SimplePlot]] = defaultdict(list)
        self._detailed_plots: Dict[str, DetailedPlot] = {}

    def create_summary_plots(self):
        """
        Constructs simple plots based on config specs
        """

        model_names = self._result_manager._profile_model_names

        for plots_key in model_names:
            self._create_summary_plot_for_model(
                plots_key=plots_key,
                model_name=plots_key,
                num_results=self._config.num_configs_per_model)

        if self._config.num_top_model_configs:
            self._create_summary_plot_for_model(
                plots_key=TOP_MODELS_REPORT_KEY,
                model_name=None,
                num_results=self._config.num_top_model_configs)

    def _create_summary_plot_for_model(self, model_name, plots_key,
                                       num_results):
        """
        helper function that creates the summary plots
        for a given model
        """

        for plot_config in self._config.plots:
            constraints = self._constraints[GLOBAL_CONSTRAINTS_KEY]
            if plots_key in self._constraints:
                constraints = self._constraints[plots_key]
            for run_config_result in self._result_manager.top_n_results(
                    model_name=model_name, n=num_results, include_default=True):
                if run_config_result.run_config().cpu_only():
                    if plot_config.y_axis() == 'gpu_used_memory':
                        plot_name, plot_config_dict = list(
                            DEFAULT_CPU_MEM_PLOT.items())[0]
                        plot_config = ConfigPlot(plot_name, **plot_config_dict)
                self._create_update_simple_plot(
                    plots_key=plots_key,
                    plot_config=plot_config,
                    run_config_measurements=run_config_result.
                    run_config_measurements(),
                    constraints=constraints)

    def _create_update_simple_plot(self, plots_key, plot_config,
                                   run_config_measurements, constraints):
        """
        Creates or updates a single simple plot, given a config name, 
        some measurements, and a key to put the plot into the simple plots
        """

        if plots_key not in self._simple_plots:
            self._simple_plots[plots_key] = {}
        if plot_config.name() not in self._simple_plots[plots_key]:
            self._simple_plots[plots_key][plot_config.name()] = SimplePlot(
                name=plot_config.name(),
                title=plot_config.title(),
                x_axis=plot_config.x_axis(),
                y_axis=plot_config.y_axis(),
                monotonic=plot_config.monotonic())

        for run_config_measurement in run_config_measurements:
            self._simple_plots[plots_key][
                plot_config.name()].add_run_config_measurement(
                    label=run_config_measurement.model_variants_name(),
                    run_config_measurement=run_config_measurement)

        # In case this plot already had lines, we want to clear and replot
        self._simple_plots[plots_key][plot_config.name()].clear()
        self._simple_plots[plots_key][plot_config.name(
        )].plot_data_and_constraints(constraints=constraints)

    def create_detailed_plots(self):
        """
        Constructs detailed plots based on
        requested config specs
        """

        # Create detailed plots
        for model in self._config.report_model_configs:
            model_config_name = model.model_config_name()
            self._detailed_plots[model_config_name] = DetailedPlot(
                f'latency_breakdown', 'Online Performance')
            model_config, run_config_measurements = self._result_manager.get_model_configs_run_config_measurements(
                model_config_name)

            # If model_config_name was present in results
            if run_config_measurements:
                for run_config_measurement in run_config_measurements:
                    self._detailed_plots[
                        model_config_name].add_run_config_measurement(
                            run_config_measurement)
                self._detailed_plots[model_config_name].plot_data()

            # Create the simple plots for the detailed reports
            for plot_config in model.plots():
                if model_config.cpu_only() and (
                        plot_config.y_axis().startswith('gpu_') or
                        plot_config.x_axis().startswith('gpu_')):
                    continue
                self._create_update_simple_plot(
                    plots_key=model_config_name,
                    plot_config=plot_config,
                    run_config_measurements=run_config_measurements,
                    constraints=None)

    def export_summary_plots(self):
        """
        write the plots to disk
        """

        simple_plot_dir = os.path.join(self._plot_export_directory, 'simple')
        for plots_key, plot_dicts in self._simple_plots.items():
            model_plot_dir = os.path.join(simple_plot_dir, plots_key)
            os.makedirs(model_plot_dir, exist_ok=True)
            for plot in plot_dicts.values():
                plot.save(model_plot_dir)

    def export_detailed_plots(self):
        """
        Write detaild plots to disk
        """

        detailed_plot_dir = os.path.join(self._plot_export_directory,
                                         'detailed')
        simple_plot_dir = os.path.join(self._plot_export_directory, 'simple')
        for model_config_name, plot in self._detailed_plots.items():
            detailed_model_config_plot_dir = os.path.join(
                detailed_plot_dir, model_config_name)
            os.makedirs(detailed_model_config_plot_dir, exist_ok=True)
            plot.save(detailed_model_config_plot_dir)

            simple_model_config_plot_dir = os.path.join(simple_plot_dir,
                                                        model_config_name)
            os.makedirs(simple_model_config_plot_dir, exist_ok=True)
            for plot in self._simple_plots[model_config_name].values():
                plot.save(simple_model_config_plot_dir)
