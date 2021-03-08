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

from .common import test_result_collector as trc
from .mocks.mock_config import MockConfig
from .mocks.mock_model_config import MockModelConfig
from .mocks.mock_client import MockTritonClientMethods
from model_analyzer.config.input.config import AnalyzerConfig
from model_analyzer.cli.cli import CLI
from model_analyzer.triton.client.grpc_client import TritonGRPCClient
from model_analyzer.config.run.run_config_generator \
    import RunConfigGenerator
from unittest.mock import mock_open, patch
import yaml


class TestRunConfigGenerator(trc.TestResultCollector):
    def _evaluate_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)
        cli.parse()
        mock_config.stop()
        return config

    def test_parameter_sweep(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file', '--model-names', 'vgg11'
        ]
        yaml_content = ''
        config = self._evaluate_config(args, yaml_content)

        mock_model_config = MockModelConfig()
        mock_model_config.start()
        mock_client = MockTritonClientMethods()
        mock_client.start()
        client = TritonGRPCClient('localhost:8000')

        # When there is not any sweep_parameter the length of
        # run_configs should be equal to the length of different
        # sweep configurations per model
        with patch('model_analyzer.triton.model.model_config.open',
                   mock_open()):
            for model in config.model_names:
                run_config_generator = RunConfigGenerator(
                    model, config, client)
                run_configs = run_config_generator.get_run_configs()
                self.assertTrue(len(run_configs) == 1)
                self.assertTrue(
                    len(run_configs[0].perf_analyzer_configs()) == 1)
        mock_model_config.stop()
        mock_client.stop()

        yaml_content = yaml.dump({
            'concurrency': [2, 3, 4],
            'batch_sizes': [4, 5, 6]
        })
        config = self._evaluate_config(args, yaml_content)
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        mock_client.start()
        with patch('model_analyzer.triton.model.model_config.open',
                   mock_open()):
            for model in config.model_names:
                run_config_generator = RunConfigGenerator(
                    model, config, client)
                run_configs = run_config_generator.get_run_configs()
                self.assertTrue(len(run_configs) == 1)
                self.assertTrue(
                    len(run_configs[0].perf_analyzer_configs()) == 9)
        mock_model_config.stop()
        mock_client.stop()

        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            instance_group:
                -
                    kind: KIND_GPU
                    count: 1
                -
                    kind: KIND_CPU
                    count: 1

"""
        config = self._evaluate_config(args, yaml_content)
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        mock_client.start()
        with patch('model_analyzer.triton.model.model_config.open',
                   mock_open()):
            for model in config.model_names:
                run_config_generator = RunConfigGenerator(
                    model, config, client)
                run_configs = run_config_generator.get_run_configs()
                self.assertTrue(len(run_configs) == 1)
                self.assertTrue(
                    len(run_configs[0].perf_analyzer_configs()) == 1)
        mock_model_config.stop()
        mock_client.stop()

        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            instance_group:
            -
                -
                    kind: KIND_GPU
                    count: 1
            -
                -
                    kind: KIND_CPU
                    count: 1

"""
        config = self._evaluate_config(args, yaml_content)
        mock_model_config = MockModelConfig()
        mock_client.start()
        mock_model_config.start()
        with patch('model_analyzer.triton.model.model_config.open',
                   mock_open()):
            for model in config.model_names:
                run_config_generator = RunConfigGenerator(
                    model, config, client)
                run_configs = run_config_generator.get_run_configs()
                self.assertTrue(len(run_configs) == 2)
                for run_config in run_configs:
                    self.assertTrue(
                        len(run_config.perf_analyzer_configs()) == 1)
        mock_model_config.stop()
        mock_client.stop()

        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            instance_group:
            -
                -
                    kind: KIND_GPU
                    count: 1
                -
                    kind: KIND_CPU
                    count: 1

"""
        config = self._evaluate_config(args, yaml_content)
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        mock_client.start()
        with patch('model_analyzer.triton.model.model_config.open',
                   mock_open()):
            for model in config.model_names:
                run_config_generator = RunConfigGenerator(
                    model, config, client)
                run_configs = run_config_generator.get_run_configs()
                self.assertTrue(len(run_configs) == 1)
                for run_config in run_configs:
                    self.assertTrue(
                        len(run_config.perf_analyzer_configs()) == 1)
        mock_model_config.stop()
        mock_client.stop()

        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            instance_group:
            -
                -
                    kind: [KIND_GPU, KIND_CPU]
                    count: [1, 2, 3]
"""
        config = self._evaluate_config(args, yaml_content)
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        mock_client.start()
        with patch('model_analyzer.triton.model.model_config.open',
                   mock_open()):
            for model in config.model_names:
                run_config_generator = RunConfigGenerator(
                    model, config, client)
                run_configs = run_config_generator.get_run_configs()
                self.assertTrue(len(run_configs) == 6)
                for run_config in run_configs:
                    self.assertTrue(
                        len(run_config.perf_analyzer_configs()) == 1)
        mock_model_config.stop()
        mock_client.stop()

        yaml_content = """
concurrency: [1, 2, 3]
batch_sizes: [2, 3, 4]
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            instance_group:
            -
                -
                    kind: [KIND_GPU, KIND_CPU]
                    count: [1, 2, 3]
"""
        config = self._evaluate_config(args, yaml_content)
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        mock_client.start()
        with patch('model_analyzer.triton.model.model_config.open',
                   mock_open()):
            for model in config.model_names:
                run_config_generator = RunConfigGenerator(
                    model, config, client)
                run_configs = run_config_generator.get_run_configs()
                self.assertTrue(len(run_configs) == 6)
                instance_groups = []
                for run_config in run_configs:
                    instance_group = run_config.model_config().get_config(
                    )['instance_group']
                    instance_groups.append(instance_group)
                    self.assertTrue(
                        len(run_config.perf_analyzer_configs()) == 9)

                expected_instance_groups = [[{
                    'count': 1,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 2,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 3,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 1,
                    'kind': 'KIND_CPU'
                }], [{
                    'count': 2,
                    'kind': 'KIND_CPU'
                }], [{
                    'count': 3,
                    'kind': 'KIND_CPU'
                }]]
                self.assertTrue(len(expected_instance_groups), instance_groups)
                for instance_group in instance_groups:
                    self.assertIn(instance_group, expected_instance_groups)
        mock_model_config.stop()
        mock_client.stop()

        yaml_content = """
concurrency: [1, 2, 3]
batch_sizes: [2, 3, 4]
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            instance_group:
            -
                kind: [KIND_GPU, KIND_CPU]
                count: [1, 2, 3]
"""
        config = self._evaluate_config(args, yaml_content)
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        mock_client.start()
        with patch('model_analyzer.triton.model.model_config.open',
                   mock_open()):
            for model in config.model_names:
                run_config_generator = RunConfigGenerator(
                    model, config, client)
                run_configs = run_config_generator.get_run_configs()
                self.assertTrue(len(run_configs) == 6)
                instance_groups = []
                for run_config in run_configs:
                    instance_group = run_config.model_config().get_config(
                    )['instance_group']
                    instance_groups.append(instance_group)
                    self.assertTrue(
                        len(run_config.perf_analyzer_configs()) == 9)

                expected_instance_groups = [[{
                    'count': 1,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 2,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 3,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 1,
                    'kind': 'KIND_CPU'
                }], [{
                    'count': 2,
                    'kind': 'KIND_CPU'
                }], [{
                    'count': 3,
                    'kind': 'KIND_CPU'
                }]]
                self.assertTrue(len(expected_instance_groups), instance_groups)
                for instance_group in instance_groups:
                    self.assertIn(instance_group, expected_instance_groups)
        mock_model_config.stop()
        mock_client.stop()

        yaml_content = """
concurrency: [1, 2, 3]
batch_sizes: [2, 3, 4]
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            dynamic_batching:
              preferred_batch_size: [ 4, 8 ]
              max_queue_delay_microseconds: 100
            instance_group:
            -
                kind: [KIND_GPU, KIND_CPU]
                count: [1, 2, 3]
"""
        config = self._evaluate_config(args, yaml_content)
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        mock_client.start()
        with patch('model_analyzer.triton.model.model_config.open',
                   mock_open()):
            for model in config.model_names:
                run_config_generator = RunConfigGenerator(
                    model, config, client)
                run_configs = run_config_generator.get_run_configs()
                self.assertTrue(len(run_configs) == 6)
                instance_groups = []
                for run_config in run_configs:
                    instance_group = run_config.model_config().get_config(
                    )['instance_group']
                    instance_groups.append(instance_group)
                    self.assertTrue(
                        len(run_config.perf_analyzer_configs()) == 9)

                expected_instance_groups = [[{
                    'count': 1,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 2,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 3,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 1,
                    'kind': 'KIND_CPU'
                }], [{
                    'count': 2,
                    'kind': 'KIND_CPU'
                }], [{
                    'count': 3,
                    'kind': 'KIND_CPU'
                }]]
                self.assertTrue(
                    len(expected_instance_groups) == len(instance_groups))
                for instance_group in instance_groups:
                    self.assertIn(instance_group, expected_instance_groups)
        mock_model_config.stop()
        mock_client.stop()

        yaml_content = """
concurrency: [1, 2, 3]
batch_sizes: [2, 3, 4]
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            dynamic_batching:
              preferred_batch_size: [[ 4, 8 ], [ 5, 6 ]]
              max_queue_delay_microseconds: [100, 200]
            instance_group:
            -
                kind: [KIND_GPU, KIND_CPU]
                count: [1, 2, 3]
"""
        config = self._evaluate_config(args, yaml_content)
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        mock_client.start()
        with patch('model_analyzer.triton.model.model_config.open',
                   mock_open()):
            for model in config.model_names:
                run_config_generator = RunConfigGenerator(
                    model, config, client)
                run_configs = run_config_generator.get_run_configs()
                self.assertTrue(len(run_configs) == 24)
                instance_groups = []
                dynamic_batchings = []
                for run_config in run_configs:
                    instance_group = run_config.model_config().get_config(
                    )['instance_group']
                    dynamic_batching = run_config.model_config().get_config(
                    )['dynamic_batching']

                    dynamic_batchings.append(dynamic_batching)
                    instance_groups.append(instance_group)
                    self.assertTrue(
                        len(run_config.perf_analyzer_configs()) == 9)

                expected_instance_groups = [[{
                    'count': 1,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 2,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 3,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 1,
                    'kind': 'KIND_CPU'
                }], [{
                    'count': 2,
                    'kind': 'KIND_CPU'
                }], [{
                    'count': 3,
                    'kind': 'KIND_CPU'
                }]]

                expected_dynamic_batchings = [{
                    'preferred_batch_size': [4, 8],
                    'max_queue_delay_microseconds':
                    '100'
                }, {
                    'preferred_batch_size': [4, 8],
                    'max_queue_delay_microseconds':
                    '200'
                }, {
                    'preferred_batch_size': [5, 6],
                    'max_queue_delay_microseconds':
                    '100'
                }, {
                    'preferred_batch_size': [5, 6],
                    'max_queue_delay_microseconds':
                    '200'
                }]
                self.assertTrue(
                    len(instance_groups) == len(expected_instance_groups) *
                    len(expected_dynamic_batchings))
                for instance_group in instance_groups:
                    self.assertIn(instance_group, expected_instance_groups)

                for dynamic_batching in dynamic_batchings:
                    self.assertIn(dynamic_batching, expected_dynamic_batchings)
        mock_model_config.stop()
        mock_client.stop()

        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            dynamic_batching:
                -
                  preferred_batch_size: [ 4, 8 ]
                  max_queue_delay_microseconds: 100
                -
                  preferred_batch_size: [ 5, 6 ]
                  max_queue_delay_microseconds: 200
            instance_group:
            -
                kind: [KIND_GPU, KIND_CPU]
                count: [1, 2, 3]
"""
        config = self._evaluate_config(args, yaml_content)
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        mock_client.start()
        with patch('model_analyzer.triton.model.model_config.open',
                   mock_open()):
            for model in config.model_names:
                run_config_generator = RunConfigGenerator(
                    model, config, client)
                run_configs = run_config_generator.get_run_configs()
                self.assertTrue(len(run_configs) == 12)
                instance_groups = []
                dynamic_batchings = []
                for run_config in run_configs:
                    instance_group = run_config.model_config().get_config(
                    )['instance_group']
                    dynamic_batching = run_config.model_config().get_config(
                    )['dynamic_batching']

                    dynamic_batchings.append(dynamic_batching)
                    instance_groups.append(instance_group)
                    self.assertTrue(
                        len(run_config.perf_analyzer_configs()) == 1)

                expected_instance_groups = [[{
                    'count': 1,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 2,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 3,
                    'kind': 'KIND_GPU'
                }], [{
                    'count': 1,
                    'kind': 'KIND_CPU'
                }], [{
                    'count': 2,
                    'kind': 'KIND_CPU'
                }], [{
                    'count': 3,
                    'kind': 'KIND_CPU'
                }]]

                expected_dynamic_batchings = [{
                    'preferred_batch_size': [4, 8],
                    'max_queue_delay_microseconds':
                    '100'
                }, {
                    'preferred_batch_size': [5, 6],
                    'max_queue_delay_microseconds':
                    '200'
                }]
                self.assertTrue(
                    len(instance_groups) == len(expected_instance_groups) *
                    len(expected_dynamic_batchings))
                for instance_group in instance_groups:
                    self.assertIn(instance_group, expected_instance_groups)

                for dynamic_batching in dynamic_batchings:
                    self.assertIn(dynamic_batching, expected_dynamic_batchings)
        mock_model_config.stop()
        mock_client.stop()
