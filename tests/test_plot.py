# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

from .common import test_result_collector as trc
from .common.test_utils import construct_measurement

from .mocks.mock_matplotlib import MockMatplotlibMethods
from model_analyzer.plots.simple_plot import SimplePlot
from model_analyzer.result.result_comparator import ResultComparator


class TestPlotMethods(trc.TestResultCollector):
    def setUp(self):
        # mocks
        self.matplotlib_mock = MockMatplotlibMethods()
        self.matplotlib_mock.start()

    def test_create_plot(self):
        # Create a plot and check for call to subplots
        SimplePlot(name='test_plot',
                   title='test_title',
                   x_axis='perf_throughput',
                   y_axis='perf_latency')

        self.matplotlib_mock.assert_called_subplots()

    def test_add_measurement(self):
        plot = SimplePlot(name='test_plot',
                          title='test_title',
                          x_axis='perf_throughput',
                          y_axis='perf_latency')

        gpu_data = {0: {'gpu_used_memory': 5000, 'gpu_utilization': 50}}
        non_gpu_data = {'perf_throughput': 200, 'perf_latency': 8000}
        objective_spec = {'perf_throughput': 10, 'perf_latency': 5}
        measurement = construct_measurement(
            'test_model', gpu_data, non_gpu_data,
            ResultComparator(metric_objectives=objective_spec))

        # Add above measurement
        plot.add_measurement('test_model_label1', measurement=measurement)
        self.assertDictEqual(
            plot.data(),
            {'test_model_label1': {
                'x_data': [200],
                'y_data': [8000]
            }})

        # Add measurment again with different label
        plot.add_measurement('test_model_label2', measurement=measurement)
        self.assertDictEqual(
            plot.data(), {
                'test_model_label1': {
                    'x_data': [200],
                    'y_data': [8000]
                },
                'test_model_label2': {
                    'x_data': [200],
                    'y_data': [8000]
                }
            })

    def test_plot_data(self):
        plot = SimplePlot(name='test_plot',
                          title='test_title',
                          x_axis='perf_throughput',
                          y_axis='perf_latency')

        gpu_data = {0: {'gpu_used_memory': 5000, 'gpu_utilization': 50}}
        non_gpu_data = {'perf_throughput': 200, 'perf_latency': 8000}
        objective_spec = {'perf_throughput': 10, 'perf_latency': 5}
        measurement = construct_measurement(
            'test_model', gpu_data, non_gpu_data,
            ResultComparator(metric_objectives=objective_spec))
        plot.add_measurement('test_model_label', measurement=measurement)

        # Call plot and assert args
        plot.plot_data_and_constraints(constraints={})
        self.matplotlib_mock.assert_called_plot_with_args(
            x_data=[200], y_data=[8000], marker='o', label='test_model_label')

    def test_save(self):
        plot = SimplePlot(name='test_plot',
                          title='test_title',
                          x_axis='perf_throughput',
                          y_axis='perf_latency')

        plot.save('test_path')
        self.matplotlib_mock.assert_called_save_with_args(
            'test_path/test_plot')

    def tearDown(self):
        self.matplotlib_mock.stop()
