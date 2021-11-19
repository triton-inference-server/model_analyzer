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
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
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

    def _evaluate_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = ConfigCommandProfile()
        cli = CLI()
        cli.add_subcommand(
            cmd='profile',
            help=
            'Run model inference profiling based on specified CLI or config options.',
            config=config)
        cli.parse()
        mock_config.stop()
        return config

    def mock_get_state_variable(self, name):
        return {
            'model': {
                'config1': None,
                'config2': None,
                'config3': None,
                'config4': None
            }
        }

    @patch(
        'model_analyzer.state.analyzer_state_manager.AnalyzerStateManager.get_state_variable',
        mock_get_state_variable)
    def test_get_num_profiled_configs(self):
        """
        Tests that the member function returning the number of profiled configs
        works correctly.
        """

        args = [
            "model-analyzer", "profile", "--model-repository", "/tmp",
            "--profile-models", "test"
        ]
        config = self._evaluate_config(args, '')
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config, None, state_manager)
        self.assertEqual(analyzer._get_num_profiled_configs(), 4)

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

    @patch('model_analyzer.result.result_manager.ResultManager.top_n_results',
           mock_top_n_results)
    def test_get_top_3_model_config_names(self):
        """
        Tests that the member function returning the top 3 model config names
        works correctly.
        """

        args = [
            "model-analyzer", "profile", "--model-repository", "/tmp",
            "--profile-models", "test"
        ]
        config = self._evaluate_config(args, '')
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config, None, state_manager)
        self.assertEqual(analyzer._get_top_3_model_config_names(),
                         ['config1', 'config3', 'config4'])
