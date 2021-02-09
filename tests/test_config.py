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
from .mocks.mock_config import MockConfig
from .common import test_result_collector as trc
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from model_analyzer.cli.cli import CLI
from model_analyzer.config.config import AnalyzerConfig
from model_analyzer.config.config_model import ConfigModel


class TestConfig(trc.TestResultCollector):

    def _evaluate_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)
        cli.parse()
        mock_config.stop()

        return config

    def test_config(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file', '--model-names', 'vgg11'
        ]
        yaml_content = 'model_repository: yaml_repository'
        config = self._evaluate_config(args, yaml_content)

        # CLI flag has the highest priority
        self.assertTrue(
            config.get_all_config()['model_repository'] == 'cli_repository')

        args = [
            'model-analyzer', '-f', 'path-to-config-file', '--model-names',
            'vgg11'
        ]
        yaml_content = 'model_repository: yaml_repository'
        config = self._evaluate_config(args, yaml_content)

        # If CLI flag doesn't exist, YAML config has the highest priority
        self.assertTrue(
            config.get_all_config()['model_repository'] == 'yaml_repository')

        args = ['model-analyzer', '-f', 'path-to-config-file']
        yaml_content = 'model_repository: yaml_repository'
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)

        # When a required field is not specified, parse will lead to an
        # exception
        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()

        mock_config.stop()

    def test_range_and_list_values(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]
        yaml_content = 'model_names: model_1,model_2'
        config = self._evaluate_config(args, yaml_content)
        model_names = ['model_1', 'model_2']
        for model_config, model_name in zip(
                config.get_all_config()['model_names'],
                model_names):
            self.assertIsInstance(model_config, ConfigModel)
            self.assertTrue(model_config.model_name() == model_name)

        yaml_content = """
model_names:
    - model_1
    - model_2
"""
        config = self._evaluate_config(args, yaml_content)
        for model_config, model_name in zip(
                config.get_all_config()['model_names'],
                model_names):
            self.assertIsInstance(model_config, ConfigModel)
            self.assertTrue(model_config.model_name() == model_name)

        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file', '--model-names', 'model_1,model_2'
        ]
        yaml_content = """
batch_sizes:
    - 2
    - 3
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertTrue(config.get_all_config()['batch_sizes'] == [2, 3])

        yaml_content = """
batch_sizes: 2
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertTrue(config.get_all_config()['batch_sizes'] == [2])

        yaml_content = """
concurrency: 2
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertTrue(config.get_all_config()['concurrency'] == [2])
        self.assertTrue(config.get_all_config()['batch_sizes'] == [1])

        yaml_content = """
batch_sizes:
    start: 2
    stop: 6
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertTrue(
            config.get_all_config()['batch_sizes'] == [2, 3, 4, 5, 6])

        yaml_content = """
batch_sizes:
    start: 2
    stop: 6
    step: 2
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertTrue(config.get_all_config()['batch_sizes'] == [2, 4, 6])

    def test_object(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
          - 1
          - 2
          - 3
          - 4
  - vgg_19_graphdef
"""
        config = self._evaluate_config(args, yaml_content)

        expected_model_objects = [ConfigModel('vgg_16_graphdef', parameters={
            'concurrency': [1, 2, 3, 4]
        }), ConfigModel('vgg_19_graphdef')]
        for model, expected_model in zip(
                config.get_all_config()['model_names'],
                expected_model_objects):
            self.assertIsInstance(model, ConfigModel)
            self.assertTrue(expected_model.model_name() == model.model_name())
            self.assertTrue(expected_model.parameters() == model.parameters())

        yaml_content = """
model_names:
  vgg_16_graphdef:
    parameters:
      concurrency:
        - 1
        - 2
        - 3
        - 4
  vgg_19_graphdef:
    parameters:
      concurrency:
        - 1
        - 2
        - 3
        - 4
      batch_sizes:
          start: 2
          stop: 6
          step: 2
"""
        expected_model_objects = [
            ConfigModel('vgg_16_graphdef', parameters={
                'concurrency': [1, 2, 3, 4]
            }),
            ConfigModel('vgg_19_graphdef',
                        parameters={'concurrency': [1, 2, 3, 4],
                                    'batch_sizes': [2, 4, 6]})]
        config = self._evaluate_config(args, yaml_content)
        for model, expected_model in zip(
                config.get_all_config()['model_names'],
                expected_model_objects):
            self.assertIsInstance(model, ConfigModel)
            self.assertTrue(expected_model.model_name() == model.model_name())
            self.assertTrue(expected_model.parameters() == model.parameters())

    def test_constraints(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
          - 1
          - 2
          - 3
          - 4
      objectives:
        - throughput
        - gpu_memory
      constraints:
        gpu_memory:
          max: 80
  - vgg_19_graphdef
"""
        config = self._evaluate_config(args, yaml_content)
        expected_model_objects = [
            ConfigModel('vgg_16_graphdef',
                        parameters={
                            'concurrency': [1, 2, 3, 4]
                        },
                        objectives=['throughput', 'gpu_memory'],
                        constraints={'gpu_memory': {
                            'max': 80,
                        }}),
            ConfigModel('vgg_19_graphdef')]
        for expected_model, model in zip(
                config.get_all_config()['model_names'],
                expected_model_objects):
            self.assertIsInstance(model, ConfigModel)
            self.assertTrue(expected_model.model_name() == model.model_name())
            self.assertTrue(expected_model.parameters() == model.parameters())
            self.assertTrue(expected_model.objectives() == model.objectives())
            self.assertTrue(expected_model.constraints()
                            == model.constraints())

        # GPU Memory shouldn't have min
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
          - 1
          - 2
          - 3
          - 4
      objectives:
        - throughput
        - gpu_memory
      constraints:
        gpu_memory:
          max: 80
          min: 45
  - vgg_19_graphdef
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)

        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()
        mock_config.stop()

    def test_validation(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]

        # end key should not be included in concurrency
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
            start: 4
            stop: 12
            end: 2
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)

        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()
            mock_config.stop()

        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]

        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
            start: 13
            stop: 12
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)

        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()
            mock_config.stop()

        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency: []
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)

        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()
            mock_config.stop()


if __name__ == '__main__':
    unittest.main()
