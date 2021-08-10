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

import os
import matplotlib.pyplot as plt
from collections import defaultdict

from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig


class SimplePlot:
    """
    A wrapper class around a matplotlib
    plot that adapts with the kinds of 
    plots the model analyzer wants to generates

    A singe plot holds data for multiple 
    model configs, but only holds one
    type of plot
    """

    def __init__(self, name, title, x_axis, y_axis, monotonic=False):
        """
        Parameters
        ----------
        name: str
            The name of the file that the plot
            will be saved as 
        title : str
            The title of this plot/figure
        x_axis : str
            The metric tag for the x-axis of this plot
        y_axis : str
            The metric tag for the y-axis of this plot
        monotonic: bool
            Whether or not to prune decreasing points in this
            plot
        """

        self._name = name
        self._title = title
        self._x_axis = x_axis
        self._y_axis = y_axis
        self._monotonic = monotonic

        self._fig, self._ax = plt.subplots()

        self._data = {}

    def add_measurement(self, model_config_label, measurement):
        """
        Adds a measurment to this plot

        Parameters
        ----------
        model_config_label : str
            The name of the model config this measurement
            is taken from. 
        measurement : Measurement
            The measurement containing the data to
            be plotted.
        """

        if model_config_label not in self._data:
            self._data[model_config_label] = defaultdict(list)

        if self._x_axis.replace('_', '-') in PerfAnalyzerConfig.allowed_keys():
            self._data[model_config_label]['x_data'].append(
                measurement.get_parameter(tag=self._x_axis.replace('_', '-')))
        else:
            self._data[model_config_label]['x_data'].append(
                measurement.get_metric_value(tag=self._x_axis))

        if self._y_axis.replace('_', '-') in PerfAnalyzerConfig.allowed_keys():
            self._data[model_config_label]['y_data'].append(
                measurement.get_parameter(tag=self._y_axis.replace('_', '-')))
        else:
            self._data[model_config_label]['y_data'].append(
                measurement.get_metric_value(tag=self._y_axis))

    def clear(self):
        """
        Clear the contents of the current Axes object
        """

        self._ax.clear()

    def plot_data_and_constraints(self, constraints):
        """
        Calls plotting function
        on this plot's Axes object

        Parameters
        ----------
        constraints: dict
            The keys are metric tags and values are dicts whose
            keys are constraint types (min, max) and values are their 
            values
        """

        self._ax.set_title(self._title)

        if self._x_axis.replace('_', '-') in PerfAnalyzerConfig.allowed_keys():
            self._x_header = self._x_axis.replace('_', ' ').title()
        else:
            self._x_header = MetricsManager.get_metric_types(
                [self._x_axis])[0].header(aggregation_tag='')

        if self._y_axis.replace('_', '-') in PerfAnalyzerConfig.allowed_keys():
            self._y_header = self._y_axis.replace('_', ' ').title()
        else:
            self._y_header = MetricsManager.get_metric_types(
                [self._y_axis])[0].header(aggregation_tag='')

        self._ax.set_xlabel(self._x_header)
        self._ax.set_ylabel(self._y_header)

        for model_config_name, data in self._data.items():
            # Sort the data by x-axis
            x_data, y_data = (
                list(t)
                for t in zip(*sorted(zip(data['x_data'], data['y_data']))))

            if self._monotonic:
                filtered_x, filtered_y = [x_data[0]], [y_data[0]]
                for i in range(1, len(x_data)):
                    if y_data[i] > filtered_y[-1]:
                        filtered_x.append(x_data[i])
                        filtered_y.append(y_data[i])
                x_data, y_data = filtered_x, filtered_y

            self._ax.plot(x_data, y_data, marker='o', label=model_config_name)

        # Plot constraints
        if constraints:
            if self._x_axis in constraints:
                for _, constraint_val in constraints[self._x_axis].items():
                    constraint_label = f"Target {self._x_header.rsplit(' ',1)[0]}"
                    self._ax.axvline(x=constraint_val,
                                     linestyle='--',
                                     label=constraint_label)
            if self._y_axis in constraints:
                for _, constraint_val in constraints[self._y_axis].items():
                    constraint_label = f"Target {self._y_header.rsplit(' ', 1)[0]}"
                    self._ax.axhline(y=constraint_val,
                                     linestyle='--',
                                     label=constraint_label)
            # plot h lines
        self._ax.legend()
        self._ax.grid()

    def data(self):
        """
        Get the data in this plot
        
        Returns
        -------
        dict
            keys are line labels
            and values are lists of floats
        """

        return self._data

    def save(self, filepath):
        """
        Saves a .png of the plot to disk

        Parameters
        ----------
        filepath : the path to the directory
            this plot should be saved to
        """

        self._fig.savefig(os.path.join(filepath, self._name))
