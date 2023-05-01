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

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.record.metrics_manager import MetricsManager

import os
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from collections import defaultdict

import logging

logging.getLogger('matplotlib').setLevel(logging.ERROR)

logger = logging.getLogger(LOGGER_NAME)


class DetailedPlot:
    """
    A wrapper class around a matplotlib
    plot that adapts with the kinds of 
    plots the model analyzer wants to generates

    Detailed plots detail th
    """

    detailed_metrics = [
        'perf_server_queue', 'perf_server_compute_input',
        'perf_server_compute_infer', 'perf_server_compute_output'
    ]

    def __init__(self, name, title, bar_width=0.5):
        """
        Parameters
        ----------
        name: str
            The name of the file that the plot
            will be saved as 
        title : str
            The title of this plot/figure
        bar_width: float
            width of the latency breakdown bars
        """

        self._name = name
        self._title = title

        self._fig, self._ax_latency = plt.subplots()
        self._ax_latency.set_title(title)
        self._ax_throughput = self._ax_latency.twinx()

        latency_axis_label, throughput_axis_label = [
            metric.header(aggregation_tag='')
            for metric in MetricsManager.get_metric_types(
                ['perf_latency_avg', 'perf_throughput'])
        ]

        self._bar_colors = {
            'perf_server_queue': '#9daecc',
            'perf_server_compute_input': '#7eb7e8',
            'perf_server_compute_infer': '#0072ce',
            'perf_server_compute_output': '#254b87',
            'perf_throughput': '#5E5E5E'
        }

        self._bar_width = bar_width
        self._legend_x = 0.92
        self._legend_y = 1.15
        self._legend_font_size = 10
        self._fig.set_figheight(8)
        self._fig.set_figwidth(12)

        self._ax_latency.set_xlabel('Concurrent Client Requests')
        self._ax_latency.set_ylabel(latency_axis_label)
        self._ax_throughput.set_ylabel(throughput_axis_label)

        self._data = defaultdict(list)

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

    def add_run_config_measurement(self, run_config_measurement):
        """
        Adds a measurement to this plot

        Parameters
        ----------
        measurement : Measurement
            The measurement containing the data to
            be plotted.
        """

        # TODO-TMA-568: This needs to be updated because there will be multiple model configs
        if 'concurrency-range' in run_config_measurement.model_specific_pa_params(
        )[0] and run_config_measurement.model_specific_pa_params(
        )[0]['concurrency-range']:
            self._data['concurrency'].append(
                run_config_measurement.model_specific_pa_params()[0]
                ['concurrency-range'])

        if 'request-rate-range' in run_config_measurement.model_specific_pa_params(
        )[0] and run_config_measurement.model_specific_pa_params(
        )[0]['request-rate-range']:
            self._data['request_rate'].append(
                run_config_measurement.model_specific_pa_params()[0]
                ['request-rate-range'])

        self._data['perf_throughput'].append(
            run_config_measurement.get_non_gpu_metric_value(
                tag='perf_throughput'))

        for metric in self.detailed_metrics:
            if MetricsManager.is_gpu_metric(tag=metric):
                self._data[metric].append(
                    run_config_measurement.get_gpu_metric_value(tag=metric))
            else:
                self._data[metric].append(
                    run_config_measurement.get_non_gpu_metric_value(tag=metric))

    def plot_data(self):
        """
        Calls plotting function
        on this plot's Axes object
        """

        # Need to change the default x-axis plot title for request rates
        if 'request_rate' in self._data and self._data['request_rate'][0]:
            self._ax_latency.set_xlabel('Client Request Rate')

        # Sort the data by request rate or concurrency
        if 'request_rate' in self._data and self._data['request_rate'][0]:
            print(f"\n\nFound request rate: {self._data['request_rate']}\n\n")
            sort_indices = list(
                zip(*sorted(enumerate(self._data['request_rate']),
                            key=lambda x: x[1])))[0]
        else:
            sort_indices = list(
                zip(*sorted(enumerate(self._data['concurrency']),
                            key=lambda x: x[1])))[0]

        sorted_data = {
            key: [data_list[i] for i in sort_indices
                 ] for key, data_list in self._data.items()
        }

        # Plot latency breakdown bars
        labels = dict(
            zip(self.detailed_metrics, [
                metric.header() for metric in MetricsManager.get_metric_types(
                    tags=self.detailed_metrics)
            ]))
        bottoms = None

        if 'request_rate' in self._data:
            sorted_data['indices'] = list(map(str, sorted_data['request_rate']))
        else:
            sorted_data['indices'] = list(map(str, sorted_data['concurrency']))

        # Plot latency breakdown with concurrency casted as string to make uniform x
        for metric, label in labels.items():
            self._ax_latency.bar(sorted_data['indices'],
                                 sorted_data[metric],
                                 width=self._bar_width,
                                 label=label,
                                 bottom=bottoms,
                                 color=self._bar_colors[metric])
            if not bottoms:
                bottoms = sorted_data[metric]
            else:
                bottoms = list(
                    map(lambda x, y: x + y, bottoms, sorted_data[metric]))

        # Plot the inference line
        inference_line = self._ax_throughput.plot(
            sorted_data['indices'],
            sorted_data['perf_throughput'],
            label='Inferences/second',
            marker='o',
            color=self._bar_colors['perf_throughput'])

        # Create legend handles
        handles = [
            mpatches.Patch(color=self._bar_colors[m], label=labels[m])
            for m in self._bar_colors
            if m != 'perf_throughput'
        ]
        handles.append(inference_line[0])

        self._ax_latency.legend(handles=handles,
                                ncol=(len(self._bar_colors) // 2) + 1,
                                bbox_to_anchor=(self._legend_x, self._legend_y),
                                prop=dict(size=self._legend_font_size))
        # Annotate inferences
        for x, y in zip(sorted_data['indices'], sorted_data['perf_throughput']):
            self._ax_throughput.annotate(
                str(round(y, 2)),
                xy=(x, y),
                textcoords="offset points",  # how to position the text
                xytext=(0, 10),  # distance from text to points (x,y)
                ha='center')

        self._ax_latency.grid()
        self._ax_latency.set_axisbelow(True)

    def save(self, filepath):
        """
        Saves a .png of the plot to disk

        Parameters
        ----------
        filepath : the path to the directory
            this plot should be saved to
        """

        self._fig.savefig(os.path.join(filepath, self._name))
