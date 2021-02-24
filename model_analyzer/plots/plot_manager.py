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

import os
from .plot import Plot


class PlotManager:
    """
    This class manages the construction and arrangement
    of plots generated by model analyzer
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : AnalyzerConfig
            The model analyzer's config containing information
            about the kind of plots to generate
        """

        self._config = config

        # Construct plot output directory
        self._plot_export_directory = os.path.join(config.export_path, 'plots')
        os.makedirs(self._plot_export_directory, exist_ok=True)

        # Add all required plots
        self._current_plots = self._new_plots()

        # List of plots per model
        self._completed_plots = {}

    def _new_plots(self):
        """
        Constructs new plots based on config
        """

        return [
            Plot(name=plot.name(),
                 title=plot.title(),
                 x_axis=plot.x_axis(),
                 y_axis=plot.y_axis()) for plot in self._config.plots
        ]

    def add_measurement(self, model_config_label, measurement):
        """
        Add a measurement to all plots
        
        Parameters
        ----------
        model_config_label : str
            Name to use to identify the line on plot
        measurement : Measurement
            The measurment to add to this 
        """

        for plot in self._current_plots:
            plot.add_measurement(model_config_label=model_config_label,
                                 measurement=measurement)

    def complete_plots(self, model_name):
        """
        Finish plotting the data
        and write the plots to disk

        Parameters
        ----------
        model_name : str
            The name of the model whose 
            plots are being finished
        """

        completed_plots = []
        for plot in self._current_plots:
            plot.plot_data()
            completed_plots.append(plot)

        self._completed_plots[model_name] = completed_plots
        self._current_plots = self._new_plots()

    def export_plots(self):
        """
        Save all plots to disk in 
        plot directory
        """

        for model_name, completed_plots in self._completed_plots.items():
            model_plot_dir = os.path.join(self._plot_export_directory,
                                          model_name)
            os.makedirs(model_plot_dir, exist_ok=True)
            for plot in completed_plots:
                plot.save(filepath=model_plot_dir)
