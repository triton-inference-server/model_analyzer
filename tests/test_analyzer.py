# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest
from unittest.mock import patch, MagicMock
from model_analyzer.analyzer import Analyzer
from model_analyzer.config.input.config_status import ConfigStatus
from model_analyzer.constants import CONFIG_PARSER_SUCCESS
from model_analyzer.result.results import Results
from model_analyzer.result.run_config_result import RunConfigResult
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.config.run.model_run_config import ModelRunConfig

from tests.common.test_utils import evaluate_mock_config

from .common import test_result_collector as trc


class TestAnalyzer(trc.TestResultCollector):
    """
    Tests the methods of the Analyzer class
    """

    def setUp(self):
        NotImplemented

    def tearDown(self):
        patch.stopall()

    def mock_get_state_variable(self, name):
        if name == 'ResultManager.results':
            return Results()
        else:
            return {
                'model1': {
                    'config1': None,
                    'config2': None,
                    'config3': None,
                    'config4': None
                }
            }

    def mock_get_list_of_models(self):
        return ['model1']

    @patch.multiple(f'{AnalyzerStateManager.__module__}.AnalyzerStateManager',
                    get_state_variable=mock_get_state_variable,
                    exiting=lambda _: False)
    @patch.multiple(f'{Analyzer.__module__}.Analyzer',
                    _create_metrics_manager=MagicMock(),
                    _create_model_manager=MagicMock(),
                    _get_server_only_metrics=MagicMock(),
                    _profile_models=MagicMock())
    def test_profile_skip_summary_reports(self, **mocks):
        """
        Tests when the skip_summary_reports config option is turned on,
        the profile stage does not create any summary reports.

        NOTE: this test only ensures that the reports are not created with
        the default export-path.
        """
        args = [
            'model-analyzer', 'profile', '--model-repository', '/tmp',
            '--profile-models', 'model1', '--config-file', '/tmp/my_config.yml',
            '--checkpoint-directory', '/tmp/my_checkpoints',
            '--skip-summary-reports'
        ]
        config = evaluate_mock_config(args, '', subcommand="profile")
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config,
                            None,
                            state_manager,
                            checkpoint_required=False)
        analyzer.profile(client=None, gpus=None, mode=None, verbose=False)

        path = os.getcwd()
        self.assertFalse(os.path.exists(os.path.join(path, "plots")))
        self.assertFalse(os.path.exists(os.path.join(path, "results")))
        self.assertFalse(os.path.exists(os.path.join(path, "reports")))

    @patch(
        'model_analyzer.state.analyzer_state_manager.AnalyzerStateManager.get_state_variable',
        mock_get_state_variable)
    @patch('model_analyzer.result.results.Results.get_list_of_models',
           mock_get_list_of_models)
    def test_get_analyze_command_help_string(self):
        """
        Tests that the member function returning the analyze command help string
        works correctly.
        """

        args = [
            'model-analyzer', 'profile', '--model-repository', '/tmp',
            '--profile-models', 'model1', '--config-file', '/tmp/my_config.yml',
            '--checkpoint-directory', '/tmp/my_checkpoints'
        ]
        config = evaluate_mock_config(args, '', subcommand="profile")
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config,
                            None,
                            state_manager,
                            checkpoint_required=False)
        self.assertEqual(
            analyzer._get_analyze_command_help_string(),
            'To analyze the profile results and find the best configurations, '
            'run `model-analyzer analyze --analysis-models model1 '
            '--config-file /tmp/my_config.yml --checkpoint-directory '
            '/tmp/my_checkpoints`')

    def mock_top_n_results(self, model_name=None, n=-1):

        rc1 = RunConfig({})
        rc1.add_model_run_config(
            ModelRunConfig(
                "fake_model_name",
                ModelConfig.create_from_dictionary({"name": "config1"}),
                MagicMock()))
        rc2 = RunConfig({})
        rc2.add_model_run_config(
            ModelRunConfig(
                "fake_model_name",
                ModelConfig.create_from_dictionary({"name": "config3"}),
                MagicMock()))
        rc3 = RunConfig({})
        rc3.add_model_run_config(
            ModelRunConfig(
                "fake_model_name",
                ModelConfig.create_from_dictionary({"name": "config4"}),
                MagicMock()))

        return [
            RunConfigResult("fake_model_name", rc1, MagicMock()),
            RunConfigResult("fake_model_name", rc2, MagicMock()),
            RunConfigResult("fake_model_name", rc3, MagicMock())
        ]

    def mock_check_for_models_in_checkpoint(self):
        return True

    @patch(
        'model_analyzer.config.input.config_command_analyze.file_path_validator',
        lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS))
    @patch(
        'model_analyzer.config.input.config_command_analyze.ConfigCommandAnalyze._preprocess_and_verify_arguments',
        lambda _: None)
    @patch('model_analyzer.result.result_manager.ResultManager.top_n_results',
           mock_top_n_results)
    @patch(
        'model_analyzer.result.result_manager.ResultManager._check_for_models_in_checkpoint',
        mock_check_for_models_in_checkpoint)
    def test_get_report_command_help_string(self):
        """
        Tests that the member function returning the report command help string
        works correctly.
        """

        args = [
            'model-analyzer', 'analyze', '--analysis-models', 'model1',
            '--config-file', '/tmp/my_config.yml', '--checkpoint-directory',
            '/tmp/my_checkpoints', '--export-path', '/tmp/my_export_path'
        ]
        config = evaluate_mock_config(args, '', subcommand="analyze")
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config,
                            None,
                            state_manager,
                            checkpoint_required=False)
        self.assertEqual(
            analyzer._get_report_command_help_string(),
            'To generate detailed reports for the 3 best configurations, run '
            '`model-analyzer report --report-model-configs '
            'config1,config3,config4 --export-path /tmp/my_export_path '
            '--config-file /tmp/my_config.yml --checkpoint-directory '
            '/tmp/my_checkpoints`')


if __name__ == '__main__':
    unittest.main()
