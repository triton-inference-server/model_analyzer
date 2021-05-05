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

from model_analyzer.config.input.config_defaults import DEFAULT_CPU_MEM_PLOT
from model_analyzer.config.input.objects.config_plot import ConfigPlot
from model_analyzer.constants import TOP_MODELS_REPORT_KEY
from model_analyzer.result.constraint_manager import ConstraintManager

from .simple_plot import SimplePlot
from .detailed_plot import DetailedPlot

import os
from collections import defaultdict


class PlotManager:
    """
    This class manages the construction and arrangement
    of plots generated by model analyzer
    """
    def __init__(self, config, result_manager):
        """
        Parameters
        ----------
        config : ConfigCommandProfile
            The model analyzer's config containing information
            about the kind of plots to generate
        result_manager : ResultManager
            instance that manages the result tables and
            adding results
        """

        self._config = config
        self._result_manager = result_manager

        # Construct plot output directory
        self._plot_export_directory = os.path.join(config.export_path, 'plots')
        os.makedirs(self._plot_export_directory, exist_ok=True)

        # Dict of list of plots
        self._simple_plots = defaultdict(list)
        self._detailed_plots = {}

    def create_summary_plots(self):
        """
        Constructs simple plots based on config specs
        """

        # Constraints should be plotted as well
        self._constraints = ConstraintManager.get_constraints_for_all_models(
            self._config)

        model_names = [
            model.model_name() for model in self._config.analysis_models
        ]

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
            constraints = self._constraints['default']
            if plots_key in self._constraints:
                constraints = self._constraints[plots_key]
            for result in self._result_manager.top_n_results(
                    model_name=model_name, n=num_results):
                if result.model_config().cpu_only():
                    if plot_config.y_axis() == 'gpu_used_memory':
                        plot_name, plot_config_dict = list(
                            DEFAULT_CPU_MEM_PLOT.items())[0]
                        plot_config = ConfigPlot(plot_name, **plot_config_dict)
                self._create_update_simple_plot(
                    plots_key=plots_key,
                    plot_config=plot_config,
                    measurements=result.measurements(),
                    constraints=constraints)

    def _create_update_simple_plot(self, plots_key, plot_config, measurements,
                                   constraints):
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

        for measurement in measurements:
            self._simple_plots[plots_key][plot_config.name()].add_measurement(
                model_config_label=measurement.perf_config()['model-name'],
                measurement=measurement)

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
            measurements = self._result_manager.get_model_config_measurements(
                model_config_name)[1]

            # If model_config_name was present in results
            if measurements:
                for measurement in measurements:
                    self._detailed_plots[model_config_name].add_measurement(
                        measurement)
                self._detailed_plots[model_config_name].plot_data()

            # Create the simple plots for the detailed reports
            for plot_config in model.plots():
                self._create_update_simple_plot(plots_key=model_config_name,
                                                plot_config=plot_config,
                                                measurements=measurements,
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

            simple_model_config_plot_dir = os.path.join(
                simple_plot_dir, model_config_name)
            os.makedirs(simple_model_config_plot_dir, exist_ok=True)
            for plot in self._simple_plots[model_config_name].values():
                plot.save(simple_model_config_plot_dir)
