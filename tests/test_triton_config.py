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

import unittest
import sys
from unittest.mock import MagicMock
sys.path.append('../common')

import test_result_collector as trc
from model_analyzer.config.config import AnalyzerConfig
from model_analyzer.cli.cli import CLI
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from .mocks.mock_config import MockConfig


class TestConfig(trc.TestResultCollector):
    def test_config(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file', '--model-names', 'vgg11'
        ]
        yaml_content = 'model_repository: yaml_repository'
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)
        cli.parse()

        # CLI flag has the highest priority
        self.assertTrue(
            config.get_all_config()['model_repository'] == 'cli_repository')
        mock_config.stop()

        args = [
            'model-analyzer', '-f', 'path-to-config-file', '--model-names',
            'vgg11'
        ]
        yaml_content = 'model_repository: yaml_repository'
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)
        cli.parse()

        # If CLI flag doesn't exist, YAML config has the highest priority
        self.assertTrue(
            config.get_all_config()['model_repository'] == 'yaml_repository')
        mock_config.stop()

        args = ['model-analyzer', '-f', 'path-to-config-file']
        yaml_content = 'model_repository: yaml_repository'
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)

        # When a required field is not specified, parse will lead to an exception
        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()

        # If CLI flag doesn't exist, YAML config has the highest priority
        self.assertTrue(
            config.get_all_config()['model_repository'] == 'yaml_repository')
        mock_config.stop()

    def test_range_values(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]
        yaml_content = 'model_names: model_1,model_2'
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)
        cli.parse()

        self.assertTrue(
            config.get_all_config()['model_names'] == ['model_1', 'model_2'])
        mock_config.stop()

        yaml_content = """
model_names:
    - model_1
    - model_2
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)
        cli.parse()
        self.assertTrue(
            config.get_all_config()['model_names'] == ['model_1', 'model_2'])
        mock_config.stop()

        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file', '--model-names', 'model_1,model_2'
        ]
        yaml_content = """
batch_sizes:
    - 2
    - 3
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)
        cli.parse()
        self.assertTrue(
            config.get_all_config()['batch_sizes'] == [2, 3])
        mock_config.stop()

        yaml_content = """
batch_sizes:
    start: 2
    end: 6
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)
        cli.parse()
        self.assertTrue(
            config.get_all_config()['batch_sizes'] == [2, 3, 4, 5])
        mock_config.stop()

        yaml_content = """
batch_sizes:
    start: 2
    end: 6
    step: 2
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)
        cli.parse()
        self.assertTrue(
            config.get_all_config()['batch_sizes'] == [2, 4])
        mock_config.stop()


if __name__ == '__main__':
    unittest.main()
