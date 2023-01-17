# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.constants import TOP_MODELS_REPORT_KEY
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig

from model_analyzer.reports.report_manager import ReportManager
from model_analyzer.result.run_config_result_comparator import RunConfigResultComparator
from model_analyzer.result.result_manager import ResultManager

from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.triton.model.model_config import ModelConfig

from model_analyzer.result.results import Results

from .mocks.mock_io import MockIOMethods
from .mocks.mock_matplotlib import MockMatplotlibMethods
from .mocks.mock_os import MockOSMethods
from .mocks.mock_json import MockJSONMethods

from .common.test_utils import construct_run_config_measurement, evaluate_mock_config, ROOT_DIR
from .common import test_result_collector as trc

import os, shutil

import unittest
from unittest.mock import MagicMock, patch


class TestEnsembleReportManagerMethods(trc.TestResultCollector):

    def _init_managers(self,
                       models="test_model",
                       num_configs_per_model=10,
                       mode='online',
                       subcommand='profile'):
        args = ["model-analyzer", subcommand, "-f", "path-to-config-file"]
        args.extend(["--checkpoint-directory", f"{ROOT_DIR}/ensemble-ckpt"])

        if subcommand == 'profile':
            args.extend(["--profile-models", models])
            args.extend(["--model-repository", "/tmp"])
        else:
            args.extend(["--report-model-configs", models])

        yaml_str = (f"""
            client_protocol: grpc
            export_path: {ROOT_DIR}
        """)
        config = evaluate_mock_config(args, yaml_str, subcommand=subcommand)
        state_manager = AnalyzerStateManager(config=config, server=None)

        state_manager.load_checkpoint(checkpoint_required=True)
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

    def setUp(self):
        NotImplemented

    def test_ensemble_summary(self):
        """
        Ensures the summary report sentence and table are accurate for a basic ensemble model (loaded from a checkpoint)
        """
        self._init_managers(models="ensemble_python_resnet50",
                            subcommand='profile')

        self.report_manager.create_summaries()

        expected_summary_sentence = 'In 68 measurements across 37 configurations, <strong>ensemble_python_resnet50_config_28</strong> ' \
        'is <strong>285%</strong> better than the default configuration at meeting the objectives, ' \
        'under the given constraints, on GPU(s) TITAN RTX.<BR><BR><strong>ensemble_python_resnet50_config_28</strong> is comprised of the following submodels: '\
        '<UL> <LI> <strong>preprocess_config_9</strong>: ' \
        '4 GPU instances with a max batch size of 8 on platform python </LI><LI> <strong>resnet50_trt_config_8</strong>: ' \
        '2 GPU instances with a max batch size of 8 on platform tensorrt_plan </LI> </UL>'

        summary_table, summary_sentence = \
            self.report_manager._build_summary_table(
                report_key="ensemble_python_resnet50",
                num_measurements=68,
                num_configurations=37,
                gpu_name="TITAN RTX",
                cpu_only=False)

        self.assertEqual(summary_sentence, expected_summary_sentence)

        # Check the first row of the table
        self.assertEqual(
            summary_table._rows[0][0],
            'ensemble_python_resnet50_config_28')  # model config name
        self.assertEqual(summary_table._rows[0][1], '(8, 8)')  # max batch size
        self.assertEqual(summary_table._rows[0][2],
                         '(Enabled, Enabled)')  # dynamic batching
        self.assertEqual(summary_table._rows[0][3],
                         '(4:GPU, 2:GPU)')  # instance count
        self.assertEqual(summary_table._rows[0][4], '1104.425')  # p99 latency
        self.assertEqual(summary_table._rows[0][5], '63.9635')  # throughput
        self.assertEqual(summary_table._rows[0][6], 2801)  # max gpu memory
        self.assertEqual(summary_table._rows[0][7], 2.7)  # GPU utilization

    def test_ensemble_summary_cpu_only(self):
        """
        Ensures the summary report sentence and table are accurate for a basic ensemble model (loaded from a checkpoint)
        when the cpu only flag is set
        """

        self._init_managers(models="ensemble_python_resnet50",
                            subcommand='profile')

        self.report_manager.create_summaries()

        expected_summary_sentence = 'In 68 measurements across 37 configurations, <strong>ensemble_python_resnet50_config_28</strong> ' \
        'is <strong>285%</strong> better than the default configuration at meeting the objectives, ' \
        'under the given constraints.<BR><BR><strong>ensemble_python_resnet50_config_28</strong> is comprised of the following submodels: ' \
        '<UL> <LI> <strong>preprocess_config_9</strong>: ' \
        '4 GPU instances with a max batch size of 8 on platform python </LI><LI> <strong>resnet50_trt_config_8</strong>: ' \
        '2 GPU instances with a max batch size of 8 on platform tensorrt_plan </LI> </UL>'

        summary_table, summary_sentence = \
            self.report_manager._build_summary_table(
                report_key="ensemble_python_resnet50",
                num_measurements=68,
                num_configurations=37,
                gpu_name="TITAN RTX",
                cpu_only=True)

        self.assertEqual(summary_sentence, expected_summary_sentence)

        # Check the first row of the table
        self.assertEqual(
            summary_table._rows[0][0],
            'ensemble_python_resnet50_config_28')  # model config name
        self.assertEqual(summary_table._rows[0][1], '8, 8')  # max batch size
        self.assertEqual(summary_table._rows[0][2],
                         '(Enabled, Enabled)')  # dynamic batching
        self.assertEqual(summary_table._rows[0][3],
                         '4:GPU, 2:GPU')  # instance count
        self.assertEqual(summary_table._rows[0][4], '1104.425')  # p99 latency
        self.assertEqual(summary_table._rows[0][5], '63.9635')  # throughput

    def test_ensemble_detailed(self):
        """
        Ensures the detailed report sentence is accurate for a basic ensemble model (loaded from a checkpoint)
        """
        self._init_managers(models="ensemble_python_resnet50_config_28",
                            subcommand='report')

        self.report_manager._add_detailed_report_data()
        self.report_manager._build_detailed_table(
            "ensemble_python_resnet50_config_28")
        detailed_sentence = self.report_manager._build_detailed_info(
            "ensemble_python_resnet50_config_28")

        expected_detailed_sentence = "<strong>ensemble_python_resnet50_config_28</strong> is comprised of the " \
            "following submodels:<LI> <strong>preprocess_config_9</strong>: " \
            "4 GPU instances with a max batch size of 8 on platform python </LI><LI> " \
            "<strong>resnet50_trt_config_8</strong>: 2 GPU instances with a max batch size of 8 " \
            "on platform tensorrt_plan </LI><br>10 measurement(s) were obtained for the model config " \
            "on GPU(s)  with total memory 0 GB."

        self.assertEqual(detailed_sentence, expected_detailed_sentence)

    def tearDown(self):
        patch.stopall()
        shutil.rmtree(f"{ROOT_DIR}/reports")
        shutil.rmtree(f"{ROOT_DIR}/plots")


if __name__ == '__main__':
    unittest.main()
