# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from model_analyzer.config.input.config_command_report \
    import ConfigCommandReport
from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig

from model_analyzer.reports.report_manager import ReportManager
from model_analyzer.result.run_config_result_comparator import RunConfigResultComparator
from model_analyzer.result.result_manager import ResultManager

from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.triton.model.model_config import ModelConfig

from .mocks.mock_config import MockConfig
from .mocks.mock_io import MockIOMethods
from .mocks.mock_matplotlib import MockMatplotlibMethods
from .mocks.mock_os import MockOSMethods
from .mocks.mock_json import MockJSONMethods

from .common.test_utils import construct_run_config_measurement
from .common import test_result_collector as trc

import unittest
from unittest.mock import MagicMock, patch

from .common.test_utils import convert_to_bytes


class TestReportManagerMethods(trc.TestResultCollector):

    def _evaluate_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config_analyze = ConfigCommandAnalyze()
        config_report = ConfigCommandReport()
        cli = CLI()
        cli.add_subcommand(
            cmd="analyze",
            help=
            "Collect and sort profiling results and generate data and summaries.",
            config=config_analyze)
        cli.add_subcommand(cmd='report',
                           help='Generate detailed reports for a single config',
                           config=config_report)
        cli.parse()
        mock_config.stop()

        ret = config_analyze if config_analyze.export_path else config_report
        return ret

    def _init_managers(self,
                       models="test_model",
                       num_configs_per_model=10,
                       mode='online',
                       subcommand='analyze'):
        args = ["model-analyzer", subcommand, "-f", "path-to-config-file"]
        if subcommand == 'analyze':
            args.extend(["--analysis-models", models])
        else:
            args.extend(["--report-model-configs", models])

        yaml_content = convert_to_bytes("""
            num_configs_per_model: """ + str(num_configs_per_model) + """
            client_protocol: grpc
            export_path: /test/export/path
            constraints:
              perf_latency_p99:
                max: 100
              gpu_used_memory:
                max: 10000
        """)
        config = self._evaluate_config(args, yaml_content)
        state_manager = AnalyzerStateManager(config=config, server=None)
        gpu_info = {
            'gpu_uuid': {
                'name': 'fake_gpu_name',
                'total_memory': 1024000000
            }
        }
        self.result_manager = ResultManager(config=config,
                                            state_manager=state_manager)
        self.report_manager = ReportManager(mode=mode,
                                            config=config,
                                            gpu_info=gpu_info,
                                            result_manager=self.result_manager)

    def _add_result_measurement(self,
                                model_config_name,
                                model_name,
                                avg_gpu_metrics,
                                avg_non_gpu_metrics,
                                result_comparator,
                                cpu_only=False):

        config_pb = self.model_config.copy()
        config_pb["name"] = model_config_name
        model_config = ModelConfig.create_from_dictionary(config_pb)
        model_config._cpu_only = cpu_only

        measurement = construct_run_config_measurement(
            model_name=model_name,
            model_config_names=[model_config_name],
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=avg_gpu_metrics,
            non_gpu_metric_values=[avg_non_gpu_metrics],
            metric_objectives=result_comparator._metric_weights,
            model_config_weights=[1])

        perf_config = PerfAnalyzerConfig()
        perf_config.update_config({'model-name': model_config_name})
        mrc = ModelRunConfig(model_name, model_config, perf_config)
        run_config = RunConfig({})
        run_config.add_model_run_config(mrc)

        self.result_manager.add_run_config_measurement(run_config, measurement)

    def setUp(self):
        self.model_config = {
            "platform": "tensorflow_graphdef",
            "instance_group": [{
                "count": 1,
                "kind": "KIND_GPU"
            }],
            "max_batch_size": 8,
            "dynamic_batching": {}
        }

        self.os_mock = MockOSMethods(mock_paths=[
            "model_analyzer.reports.report_manager",
            "model_analyzer.config.input.config_command_analyze",
            "model_analyzer.state.analyzer_state_manager",
            "model_analyzer.config.input.config_utils"
        ])
        self.os_mock.start()
        # Required patch ordering here
        # html_report must be patched before pdf_report
        # Likely due to patching dealing with parent + child classes
        self.io_mock = MockIOMethods(mock_paths=[
            "model_analyzer.reports.html_report",
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
            result_comparator = RunConfigResultComparator(
                metric_objectives_list=[{
                    "perf_throughput": 10
                }])

            avg_gpu_metrics = {
                0: {
                    "gpu_used_memory": 6000,
                    "gpu_utilization": 60
                }
            }

            for i in range(10):
                avg_non_gpu_metrics = {
                    "perf_throughput": 100 + 10 * i,
                    "perf_latency_p99": 4000,
                    "cpu_used_ram": 1000
                }
                self._add_result_measurement(f"test_model1_report_{i}",
                                             "test_model1", avg_gpu_metrics,
                                             avg_non_gpu_metrics,
                                             result_comparator)

            for i in range(5):
                avg_non_gpu_metrics = {
                    "perf_throughput": 200 + 10 * i,
                    "perf_latency_p99": 4000,
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
        for mode in ['offline', 'online']:
            for cpu_only in [True, False]:
                self.subtest_build_summary_table(mode, cpu_only)

    def subtest_build_summary_table(self, mode, cpu_only):
        self._init_managers(mode=mode)
        result_comparator = RunConfigResultComparator(metric_objectives_list=[{
            "perf_throughput": 10
        }])

        avg_gpu_metrics = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}

        for i in range(10, 0, -1):
            avg_non_gpu_metrics = {
                "perf_throughput": 100 + 10 * i,
                "perf_latency_p99": 4000,
                "cpu_used_ram": 1000
            }
            self._add_result_measurement(f"test_model_config_{i}", "test_model",
                                         avg_gpu_metrics, avg_non_gpu_metrics,
                                         result_comparator, cpu_only)

        self.result_manager.compile_and_sort_results()
        self.report_manager.create_summaries()

        summary_table, summary_sentence = \
            self.report_manager._build_summary_table(
                report_key="test_model",
                num_measurements=10,
                gpu_name="TITAN RTX")

        expected_summary_sentence = (
            "In 10 measurement(s), config test_model_config_10 (1/GPU model instance(s)"
            " with max batch size of 8 and dynamic batching enabled) on"
            " platform tensorflow_graphdef delivers maximum"
            " throughput under the given constraints")
        if not cpu_only:
            expected_summary_sentence += " on GPU(s) TITAN RTX"
        expected_summary_sentence += "."
        self.assertEqual(expected_summary_sentence, summary_sentence)

        # Get throughput index and make sure results are sorted
        throughput_index = summary_table.headers().index(
            "Throughput (infer/sec)")
        model_name_index = summary_table.headers().index("Model Config Name")
        for i in range(9):
            current_row = summary_table.get_row_by_index(i)
            next_row = summary_table.get_row_by_index(i + 1)
            self.assertEqual(current_row[model_name_index],
                             f"test_model_config_{10-i}")
            self.assertGreaterEqual(current_row[throughput_index],
                                    next_row[throughput_index])

    def test_build_detailed_info(self):
        for cpu_only in [True, False]:
            self._subtest_build_detailed_info(cpu_only)

    def _subtest_build_detailed_info(self, cpu_only):
        self._init_managers(models="test_model_config_10", subcommand="report")

        result_comparator = RunConfigResultComparator(metric_objectives_list=[{
            "perf_throughput": 10
        }])

        avg_gpu_metrics = {
            "gpu_uuid": {
                "gpu_used_memory": 6000,
                "gpu_utilization": 60
            }
        }

        for i in range(10, 0, -1):
            avg_non_gpu_metrics = {
                "perf_throughput": 100 + 10 * i,
                "perf_latency_p99": 4000,
                "cpu_used_ram": 1000
            }
            self._add_result_measurement(f"test_model_config_{i}",
                                         "test_model",
                                         avg_gpu_metrics,
                                         avg_non_gpu_metrics,
                                         result_comparator,
                                         cpu_only=cpu_only)

        self.report_manager._add_detailed_report_data()
        self.report_manager._build_detailed_table("test_model_config_10")
        sentence = self.report_manager._build_detailed_info(
            "test_model_config_10")

        if cpu_only:
            expected_sentence = (
                f"The model config \"test_model_config_10\" uses 1 GPU instance(s) with "
                f"a max batch size of 8 and has dynamic batching enabled. 1 measurement(s) "
                f"were obtained for the model config on CPU. "
                f"This model uses the platform tensorflow_graphdef.")
        else:
            expected_sentence = (
                f"The model config \"test_model_config_10\" uses 1 GPU instance(s) with "
                f"a max batch size of 8 and has dynamic batching enabled. 1 measurement(s) "
                f"were obtained for the model config on GPU(s) fake_gpu_name with memory limit(s) 1.0 GB. "
                f"This model uses the platform tensorflow_graphdef.")

        self.assertEqual(expected_sentence, sentence)

    @patch(
        'model_analyzer.plots.plot_manager.PlotManager._create_update_simple_plot'
    )
    @patch('model_analyzer.result.result_table.ResultTable.insert_row_by_index')
    def test_summary_default_within_top(self, add_table_fn, add_plot_fn):
        '''
        Test summary report generation when default is in the top n configs
        
        Creates some results where the default config is within the top n configs, 
        and then confirms that the number of entries added to plots and tables
        is correct
        '''

        default_within_top = True
        top_n = 3
        self._test_summary_counts(add_table_fn, add_plot_fn, default_within_top,
                                  top_n)

    @patch(
        'model_analyzer.plots.plot_manager.PlotManager._create_update_simple_plot'
    )
    @patch('model_analyzer.result.result_table.ResultTable.insert_row_by_index')
    def test_summary_default_not_within_top(self, add_table_fn, add_plot_fn):
        '''
        Test summary report generation when default is not in the top n configs
        
        Creates some results where the default config is not within the top n configs, 
        and then confirms that the number of entries added to plots and tables
        is correct such that it includes the default config data
        '''
        default_within_top = False
        top_n = 3
        self._test_summary_counts(add_table_fn, add_plot_fn, default_within_top,
                                  top_n)

    def _test_summary_counts(self, add_table_fn, add_plot_fn,
                             default_within_top, top_n):
        '''
        Helper function to test creating summary reports and confirming that the number
        of entries added to plots and tables is as expected
        '''
        num_plots_in_summary_report = 2
        num_tables_in_summary_report = 1
        expected_config_count = top_n + 1 if not default_within_top else top_n
        expected_plot_count = num_plots_in_summary_report * expected_config_count
        expected_table_count = num_tables_in_summary_report * expected_config_count

        self._init_managers("test_model1", num_configs_per_model=top_n)
        result_comparator = RunConfigResultComparator(metric_objectives_list=[{
            "perf_throughput": 10
        }])
        avg_gpu_metrics = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}
        for i in range(10):
            p99 = 20 + i
            throughput = 100 - 10 * i if default_within_top else 100 + 10 * i
            avg_non_gpu_metrics = {
                "perf_throughput": throughput,
                "perf_latency_p99": p99,
                "cpu_used_ram": 1000
            }
            name = f"test_model1_config_{i}"
            if not i:
                name = f"test_model1_config_default"
            self._add_result_measurement(name, "test_model1", avg_gpu_metrics,
                                         avg_non_gpu_metrics, result_comparator)
        self.result_manager.compile_and_sort_results()
        self.report_manager.create_summaries()

        self.assertEqual(expected_plot_count, add_plot_fn.call_count)
        self.assertEqual(expected_table_count, add_table_fn.call_count)

    def tearDown(self):
        self.matplotlib_mock.stop()
        self.io_mock.stop()
        self.os_mock.stop()
        self.json_mock.stop()


if __name__ == '__main__':
    unittest.main()
