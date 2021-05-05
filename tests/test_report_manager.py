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

from model_analyzer.config.input.config_command_analyze \
    import ConfigCommandAnalyze
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.reports.report_manager import ReportManager
from model_analyzer.result.result_comparator import ResultComparator
from model_analyzer.cli.cli import CLI

from .common import test_result_collector as trc
from .common.test_utils import construct_result

from .mocks.mock_config import MockConfig
from .mocks.mock_os import MockOSMethods
from .mocks.mock_model_config import MockModelConfig
import unittest


@unittest.skip("Under Construction")
class TestReportManagerMethods(trc.TestResultCollector):
    def _evaluate_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = ConfigCommandAnalyze()
        cli = CLI()
        cli.add_subcommand(
            cmd='analyze',
            help=
            'Collect and sort profiling results and generate data and summaries.',
            config=config)
        cli.parse()
        mock_config.stop()
        return config

    def setUp(self):
        self.model_config = {
            'platform': 'tensorflow_graphdef',
            'instance_group': [{
                'count': 1,
                'kind': 'KIND_GPU'
            }],
            'dynamic_batching': {
                'preferred_batch_size': [4, 8],
            }
        }
        mock_paths = [
            'model_analyzer.reports.report_manager',
            'model_analyzer.config.input.config_command_analyze'
        ]
        self.os_mock = MockOSMethods(mock_paths=mock_paths)
        self.os_mock.start()
        args = [
            'model-analyzer', 'analyze', '-f', 'path-to-config-file',
            '--analysis-models', 'test_model'
        ]

        yaml_content = """
            num_configs_per_model: 5
            client_protocol: grpc
            export_path: /test/export/path
            constraints:
              perf_latency:
                max: 100
        """
        config = self._evaluate_config(args, yaml_content)
        self.report_manager = ReportManager(config=config)

    def test_add_results(self):
        objective_spec = {'perf_throughput': 10}
        self.result_comparator = ResultComparator(
            metric_objectives=objective_spec)

        avg_gpu_metrics = {0: {'gpu_used_memory': 6000, 'gpu_utilization': 60}}

        for i in range(10):
            avg_non_gpu_metrics = {
                'perf_throughput': 100 + 10 * i,
                'perf_latency': 4000
            }
            self.report_manager.add_result(
                report_key='test_report1',
                result=construct_result(
                    avg_gpu_metric_values=avg_gpu_metrics,
                    avg_non_gpu_metric_values=avg_non_gpu_metrics,
                    comparator=self.result_comparator))

        for i in range(5):
            avg_non_gpu_metrics = {
                'perf_throughput': 200 + 10 * i,
                'perf_latency': 4000
            }
            self.report_manager.add_result(
                report_key='test_report2',
                result=construct_result(
                    avg_gpu_metric_values=avg_gpu_metrics,
                    avg_non_gpu_metric_values=avg_non_gpu_metrics,
                    comparator=self.result_comparator))

        self.assertEqual(self.report_manager.report_keys(),
                         ['test_report1', 'test_report2'])
        report1_data = self.report_manager.data('test_report1')
        report2_data = self.report_manager.data('test_report2')

        self.assertEqual(len(report1_data), 10)
        self.assertEqual(len(report2_data), 5)

    def test_build_summary_table(self):
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        objective_spec = {'perf_throughput': 10}
        self.result_comparator = ResultComparator(
            metric_objectives=objective_spec)

        avg_gpu_metrics = {0: {'gpu_used_memory': 6000, 'gpu_utilization': 60}}

        for i in range(10, 0, -1):
            avg_non_gpu_metrics = {
                'perf_throughput': 100 + 10 * i,
                'perf_latency': 4000,
                'cpu_used_ram': 1000
            }
            self.model_config['name'] = f'model_{i}'
            model_config = ModelConfig.create_from_dictionary(
                self.model_config)
            self.report_manager.add_result(
                report_key='test_report',
                result=construct_result(
                    avg_gpu_metric_values=avg_gpu_metrics,
                    avg_non_gpu_metric_values=avg_non_gpu_metrics,
                    comparator=self.result_comparator,
                    model_config=model_config))

        summary_table, summary_sentence = \
            self.report_manager._build_summary_table(
                report_key='test_report',
                num_measurements=10,
                gpu_name='TITAN RTX')

        expected_summary_sentence = (
            "In 10 measurement(s), 1/GPU model instance(s)"
            " with preferred batch size of [4 8] on"
            " platform tensorflow_graphdef delivers maximum"
            " throughput under the given constraints on GPU(s) TITAN RTX.")
        self.assertEqual(expected_summary_sentence, summary_sentence)

        # Get throughput index and make sure results are sorted
        throughput_index = summary_table.headers().index(
            'Throughput (infer/sec)')
        model_name_index = summary_table.headers().index('Model Config Name')
        for i in range(9):
            current_row = summary_table.get_row_by_index(i)
            next_row = summary_table.get_row_by_index(i + 1)
            self.assertEqual(current_row[model_name_index], f'model_{10-i}')
            self.assertGreaterEqual(current_row[throughput_index],
                                    next_row[throughput_index])

    def tearDown(self):
        self.os_mock.stop()


if __name__ == '__main__':
    unittest.main()
