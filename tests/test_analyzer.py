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

from unittest.mock import patch
from model_analyzer.analyzer import Analyzer
from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_analyze import ConfigCommandAnalyze
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_status import ConfigStatus
from model_analyzer.constants import CONFIG_PARSER_SUCCESS
from model_analyzer.result.model_result import ModelResult
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.triton.model.model_config import ModelConfig

from google.protobuf import json_format
from tritonclient.grpc import model_config_pb2

from .common import test_result_collector as trc

from .mocks.mock_config import MockConfig


class TestAnalyzer(trc.TestResultCollector):
    """
    Tests the methods of the Analyzer class
    """

    def mock_get_state_variable(self, name):
        return {
            'model1': {
                'config1': None,
                'config2': None,
                'config3': None,
                'config4': None
            }
        }

    @patch(
        'model_analyzer.state.analyzer_state_manager.AnalyzerStateManager.get_state_variable',
        mock_get_state_variable)
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
        config = self._evaluate_profile_config(args, '')
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config, None, state_manager)
        self.assertEqual(
            analyzer._get_analyze_command_help_string(),
            ('To analyze the profile results and find the best configurations, '
             'run `model-analyzer analyze --analysis-models model1 '
             '--config-file /tmp/my_config.yml --checkpoint-directory '
             '/tmp/my_checkpoints`'))

    def mock_top_n_results(self, model_name=None, n=-1):
        return [
            ModelResult(
                None,
                ModelConfig(
                    json_format.ParseDict({'name': 'config1'},
                                          model_config_pb2.ModelConfig())),
                None),
            ModelResult(
                None,
                ModelConfig(
                    json_format.ParseDict({'name': 'config3'},
                                          model_config_pb2.ModelConfig())),
                None),
            ModelResult(
                None,
                ModelConfig(
                    json_format.ParseDict({'name': 'config4'},
                                          model_config_pb2.ModelConfig())),
                None)
        ]

    @patch(
        'model_analyzer.config.input.config_command_analyze.file_path_validator',
        lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS))
    @patch(
        'model_analyzer.config.input.config_command_analyze.ConfigCommandAnalyze._preprocess_and_verify_arguments',
        lambda _: None)
    @patch('model_analyzer.result.result_manager.ResultManager.top_n_results',
           mock_top_n_results)
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
        config = self._evaluate_analyze_config(args, '')
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config, None, state_manager)
        self.assertEqual(
            analyzer._get_report_command_help_string(),
            ('To generate detailed reports for the 3 best configurations, run '
             '`model-analyzer report --report-model-configs '
             'config1,config3,config4 --export-path /tmp/my_export_path '
             '--config-file /tmp/my_config.yml --checkpoint-directory '
             '/tmp/my_checkpoints`'))

    def _evaluate_profile_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = ConfigCommandProfile()
        cli = CLI()
        cli.add_subcommand(
            cmd='profile',
            help='Run model inference profiling based on specified CLI or '
            'config options.',
            config=config)
        cli.parse()
        mock_config.stop()
        return config

    def _evaluate_analyze_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = ConfigCommandAnalyze()
        cli = CLI()
        cli.add_subcommand(
            cmd='analyze',
            help='Collect and sort profiling results and generate data and '
            'summaries.',
            config=config)
        cli.parse()
        mock_config.stop()
        return config
