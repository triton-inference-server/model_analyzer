# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from .mocks.mock_config import MockConfig
from .mocks.mock_os import MockOSMethods

from unittest.mock import MagicMock
from unittest.mock import patch

from google.protobuf import json_format
from tritonclient.grpc import model_config_pb2

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_profile \
    import ConfigCommandProfile
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.triton.model.model_config import ModelConfig


class TestMetricsManager(trc.TestResultCollector):

    @patch('model_analyzer.record.metrics_manager.MetricsManager.__init__',
           return_value=None)
    def test_execute_run_config(self, mock_metrics_manager_init):
        """
        Tests that something something.
        """

        args = [
            'model-analyzer', 'profile', '--model-repository', 'cli_repository',
            '--profile-models', 'profile_models', '--triton-launch-mode',
            'local'
        ]
        config = self._evaluate_config(args, None)
        metrics_manager = MetricsManager()
        metrics_manager._create_model_variant = MagicMock()
        metrics_manager._get_measurement_if_config_duplicate = MagicMock(
            return_value=None)
        metrics_manager._server = MagicMock()
        metrics_manager._config = config
        metrics_manager._client = MagicMock()
        model_configs = [
            ModelConfig(
                json_format.ParseDict({'name': 'model_config_1'},
                                      model_config_pb2.ModelConfig())),
            ModelConfig(
                json_format.ParseDict({'name': 'model_config_1'},
                                      model_config_pb2.ModelConfig()))
        ]
        run_config = RunConfig(model_name=None,
                               model_configs=model_configs,
                               perf_config=None,
                               triton_env=None)

        metrics_manager.execute_run_config(run_config)

        self.assertTrue(metrics_manager._create_model_variant.call_count == 2)
        self.assertTrue(metrics_manager._client.load_model.call_count == 2)

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

    def setUp(self):
        # Mock path validation
        self.mock_os = MockOSMethods(
            mock_paths=['model_analyzer.config.input.config_utils'])
        self.mock_os.start()

    def tearDown(self):
        self.mock_os.stop()
