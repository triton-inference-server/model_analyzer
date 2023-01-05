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

from .common.test_utils import construct_run_config_measurement, evaluate_mock_config
from .common import test_result_collector as trc

import os

import unittest
from unittest.mock import MagicMock, patch


class TestEnsembleReportManagerMethods(trc.TestResultCollector):

    def _init_managers(self,
                       models="test_model",
                       num_configs_per_model=10,
                       mode='online',
                       subcommand='profile'):
        args = ["model-analyzer", subcommand, "-f", "path-to-config-file"]
        if subcommand == 'profile':
            args.extend(["--profile-models", models])
            args.extend(["--model-repository", "/tmp"])
            args.extend(["--checkpoint-directory", "/swdev/testing"])
        else:
            args.extend(["--report-model-configs", models])

        yaml_str = ("""
            client_protocol: grpc
            export_path: /swdev/testing
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

    def test_ensemble(self):
        self._init_managers(models="ensemble_python_resnet50",
                            subcommand='profile')

        self.report_manager.create_summaries()

        summary_table, summary_sentence = \
            self.report_manager._build_summary_table(
                report_key="ensemble_python_resnet50",
                num_measurements=55,
                num_configurations=26,
                gpu_name="TITAN RTX",
                cpu_only=False)

        print("Made it here")

    def tearDown(self):
        patch.stopall()


if __name__ == '__main__':
    unittest.main()