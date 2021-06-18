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

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_analyze \
    import ConfigCommandAnalyze
from model_analyzer.config.run.run_config import RunConfig

from model_analyzer.reports.report_manager import ReportManager
from model_analyzer.result.result_comparator import ResultComparator
from model_analyzer.result.result_manager import ResultManager

from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.triton.model.model_config import ModelConfig

from .mocks.mock_config import MockConfig
from .mocks.mock_io import MockIOMethods
from .mocks.mock_matplotlib import MockMatplotlibMethods
from .mocks.mock_os import MockOSMethods
from .mocks.mock_json import MockJSONMethods

from .common.test_utils import construct_measurement
from .common import test_result_collector as trc

import unittest


class TestReportManagerMethods(trc.TestResultCollector):
    def _evaluate_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = ConfigCommandAnalyze()
        cli = CLI()
        cli.add_subcommand(
            cmd="analyze",
            help=
            "Collect and sort profiling results and generate data and summaries.",
            config=config)
        cli.parse()
        mock_config.stop()
        return config

    def _init_managers(self,
                       analysis_models="test_model",
                       num_configs_per_model=10,
                       mode='online'):
        args = [
            "model-analyzer", "analyze", "-f", "path-to-config-file",
            "--analysis-models", analysis_models
        ]
        yaml_content = """
            num_configs_per_model: """ + str(num_configs_per_model) + """
            client_protocol: grpc
            export_path: /test/export/path
            constraints:
              perf_latency:
                max: 100
        """
        config = self._evaluate_config(args, yaml_content)
        state_manager = AnalyzerStateManager(config=config)
        gpu_info = {'gpu_uuid': {'name': 'gpu_name', 'total_memory': 10}}
        self.result_manager = ResultManager(config=config,
                                            state_manager=state_manager)
        self.report_manager = ReportManager(mode=mode,
                                            config=config,
                                            gpu_info=gpu_info,
                                            result_manager=self.result_manager)

    def _add_result_measurement(self, model_config_name, model_name,
                                avg_gpu_metrics, avg_non_gpu_metrics,
                                result_comparator):
        self.model_config["name"] = model_config_name
        measurement = construct_measurement(model_name, avg_gpu_metrics,
                                            avg_non_gpu_metrics,
                                            result_comparator)
        run_config = RunConfig(
            model_name, ModelConfig.create_from_dictionary(self.model_config),
            measurement.perf_config())
        self.result_manager.add_measurement(run_config, measurement)

    def setUp(self):
        self.model_config = {
            "platform": "tensorflow_graphdef",
            "instance_group": [{
                "count": 1,
                "kind": "KIND_GPU"
            }],
            "dynamic_batching": {
                "preferred_batch_size": [4, 8],
            }
        }

        self.os_mock = MockOSMethods(mock_paths=[
            "model_analyzer.reports.report_manager",
            "model_analyzer.config.input.config_command_analyze",
            "model_analyzer.state.analyzer_state_manager"
        ])
        self.os_mock.start()
        self.io_mock = MockIOMethods(mock_paths=[
            "model_analyzer.reports.pdf_report",
            "model_analyzer.state.analyzer_state_manager"
        ],
                                     read_data=[bytes(">:(".encode("ascii"))])
        self.io_mock.start()
        self.matplotlib_mock = MockMatplotlibMethods()
        self.matplotlib_mock.start()
        self.json_mock = MockJSONMethods()
        self.json_mock.start()

    def test_add_results(self):
        for mode in ['online', 'offline']:
            self._init_managers("test_model1,test_model2", mode=mode)
            result_comparator = ResultComparator(
                metric_objectives={"perf_throughput": 10})

            avg_gpu_metrics = {
                0: {
                    "gpu_used_memory": 6000,
                    "gpu_utilization": 60
                }
            }

            for i in range(10):
                avg_non_gpu_metrics = {
                    "perf_throughput": 100 + 10 * i,
                    "perf_latency": 4000,
                    "cpu_used_ram": 1000
                }
                self._add_result_measurement(f"test_model1_report_{i}",
                                             "test_model1", avg_gpu_metrics,
                                             avg_non_gpu_metrics,
                                             result_comparator)

            for i in range(5):
                avg_non_gpu_metrics = {
                    "perf_throughput": 200 + 10 * i,
                    "perf_latency": 4000,
                    "cpu_used_ram": 1000
                }
                self._add_result_measurement(f"test_model2_report_{i}",
                                             "test_model2", avg_gpu_metrics,
                                             avg_non_gpu_metrics,
                                             result_comparator)

            self.result_manager.compile_and_sort_results()
            self.report_manager.create_summaries()
            self.assertEqual(self.report_manager.report_keys(),
                             ["test_model1", "test_model2"])

            report1_data = self.report_manager.data("test_model1")
            report2_data = self.report_manager.data("test_model2")

            self.assertEqual(len(report1_data), 10)
            self.assertEqual(len(report2_data), 5)

    def test_build_summary_table(self):
        for mode in ['online', 'offline']:
            self._init_managers(mode=mode)
            result_comparator = ResultComparator(
                metric_objectives={"perf_throughput": 10})

            avg_gpu_metrics = {
                0: {
                    "gpu_used_memory": 6000,
                    "gpu_utilization": 60
                }
            }

            for i in range(10, 0, -1):
                avg_non_gpu_metrics = {
                    "perf_throughput": 100 + 10 * i,
                    "perf_latency": 4000,
                    "cpu_used_ram": 1000
                }
                self._add_result_measurement(f"model_{i}", "test_model",
                                             avg_gpu_metrics,
                                             avg_non_gpu_metrics,
                                             result_comparator)

            self.result_manager.compile_and_sort_results()
            self.report_manager.create_summaries()

            summary_table, summary_sentence = \
                self.report_manager._build_summary_table(
                    report_key="test_model",
                    num_measurements=10,
                    gpu_name="TITAN RTX")

            expected_summary_sentence = (
                "In 10 measurement(s), 1/GPU model instance(s)"
                " with preferred batch size of [4 8] on"
                " platform tensorflow_graphdef delivers maximum"
                " throughput under the given constraints on GPU(s) TITAN RTX.")
            self.assertEqual(expected_summary_sentence, summary_sentence)

            # Get throughput index and make sure results are sorted
            throughput_index = summary_table.headers().index(
                "Throughput (infer/sec)")
            model_name_index = summary_table.headers().index(
                "Model Config Name")
            for i in range(9):
                current_row = summary_table.get_row_by_index(i)
                next_row = summary_table.get_row_by_index(i + 1)
                self.assertEqual(current_row[model_name_index],
                                 f"model_{10-i}")
                self.assertGreaterEqual(current_row[throughput_index],
                                        next_row[throughput_index])

    def tearDown(self):
        self.matplotlib_mock.stop()
        self.io_mock.stop()
        self.os_mock.stop()
        self.json_mock.stop()


if __name__ == '__main__':
    unittest.main()
