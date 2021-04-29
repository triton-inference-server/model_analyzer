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
from model_analyzer.config.input.config_command_profile \
    import ConfigCommandProfile
from model_analyzer.cli.cli import CLI
from model_analyzer.triton.client.grpc_client import TritonGRPCClient
from model_analyzer.config.run.run_config_generator \
    import RunConfigGenerator


class TestRunConfigGenerator(trc.TestResultCollector):
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
        self.mock_model_config = MockModelConfig()
        self.mock_model_config.start()
        self.mock_client = MockTritonClientMethods()
        self.mock_client.start()
        self.client = TritonGRPCClient('localhost:8000')

    def test_generate_model_config_combinations(self):
        args = [
            'model-analyzer', 'profile', '--model-repository',
            'cli_repository', '-f', 'path-to-config-file', '--profile-models',
            'vgg11'
        ]

        # Empty yaml
        yaml_content = ''
        config = self._evaluate_config(args, yaml_content)
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        self.assertEqual([None], model_configs)

        # Use yaml model names
        args = [
            'model-analyzer', 'profile', '--model-repository',
            'cli_repository', '-f', 'path-to-config-file'
        ]

        # List of instance groups
        yaml_content = """
            profile_models:
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
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        expected_model_configs = [{
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 1
            }, {
                'kind': 'KIND_CPU',
                'count': 1
            }]
        }]
        self.assertEqual(expected_model_configs, model_configs)

        yaml_content = """
            profile_models:
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
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        expected_model_configs = [{
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 1
            }]
        }, {
            'instance_group': [{
                'kind': 'KIND_CPU',
                'count': 1
            }]
        }]
        self.assertEqual(expected_model_configs, model_configs)

        yaml_content = """
            profile_models:
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
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        expected_model_configs = [{
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 1
            }, {
                'kind': 'KIND_CPU',
                'count': 1
            }]
        }]
        self.assertEqual(expected_model_configs, model_configs)

        yaml_content = """
            profile_models:
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
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        expected_model_configs = [{
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 1
            }]
        }, {
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 2
            }]
        }, {
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 3
            }]
        }, {
            'instance_group': [{
                'kind': 'KIND_CPU',
                'count': 1
            }]
        }, {
            'instance_group': [{
                'kind': 'KIND_CPU',
                'count': 2
            }]
        }, {
            'instance_group': [{
                'kind': 'KIND_CPU',
                'count': 3
            }]
        }]
        self.assertEqual(expected_model_configs, model_configs)

        yaml_content = """
            profile_models:
            -
                vgg_16_graphdef:
                    model_config_parameters:
                        instance_group:
                        -
                            kind: [KIND_GPU, KIND_CPU]
                            count: [1, 2, 3]
            """
        config = self._evaluate_config(args, yaml_content)
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        self.assertEqual(expected_model_configs, model_configs)

        yaml_content = """
            profile_models:
            -
                vgg_16_graphdef:
                    model_config_parameters:
                        dynamic_batching:
                            preferred_batch_size: [ 4, 8 ]
                            max_queue_delay_microseconds: 100
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
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        expected_model_configs = [{
            'dynamic_batching': {
                'preferred_batch_size': [4, 8],
                'max_queue_delay_microseconds': 100
            },
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 1
            }]
        }, {
            'dynamic_batching': {
                'preferred_batch_size': [4, 8],
                'max_queue_delay_microseconds': 100
            },
            'instance_group': [{
                'kind': 'KIND_CPU',
                'count': 1
            }]
        }]
        self.assertEqual(expected_model_configs, model_configs)

        yaml_content = """
            profile_models:
            -
                vgg_16_graphdef:
                    model_config_parameters:
                        dynamic_batching:
                            preferred_batch_size: [[ 4, 8 ], [ 5, 6 ]]
                            max_queue_delay_microseconds: [100, 200]
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: 1
            """
        config = self._evaluate_config(args, yaml_content)
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        expected_model_configs = [{
            'dynamic_batching': {
                'preferred_batch_size': [4, 8],
                'max_queue_delay_microseconds': 100
            },
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 1
            }]
        }, {
            'dynamic_batching': {
                'preferred_batch_size': [4, 8],
                'max_queue_delay_microseconds': 200
            },
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 1
            }]
        }, {
            'dynamic_batching': {
                'preferred_batch_size': [5, 6],
                'max_queue_delay_microseconds': 100
            },
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 1
            }]
        }, {
            'dynamic_batching': {
                'preferred_batch_size': [5, 6],
                'max_queue_delay_microseconds': 200
            },
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 1
            }]
        }]
        self.assertEqual(expected_model_configs, model_configs)

        # list under dynamic batching
        yaml_content = """
            profile_models:
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
                            count: [1, 2]
            """
        config = self._evaluate_config(args, yaml_content)
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        self.assertEqual(len(model_configs), 8)
        expected_model_configs = [{
            'dynamic_batching': {
                'preferred_batch_size': [4, 8],
                'max_queue_delay_microseconds': 100
            },
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 1
            }]
        }, {
            'dynamic_batching': {
                'preferred_batch_size': [4, 8],
                'max_queue_delay_microseconds': 100
            },
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 2
            }]
        }, {
            'dynamic_batching': {
                'preferred_batch_size': [4, 8],
                'max_queue_delay_microseconds': 100
            },
            'instance_group': [{
                'kind': 'KIND_CPU',
                'count': 1
            }]
        }, {
            'dynamic_batching': {
                'preferred_batch_size': [4, 8],
                'max_queue_delay_microseconds': 100
            },
            'instance_group': [{
                'kind': 'KIND_CPU',
                'count': 2
            }]
        }, {
            'dynamic_batching': {
                'preferred_batch_size': [5, 6],
                'max_queue_delay_microseconds': 200
            },
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 1
            }]
        }, {
            'dynamic_batching': {
                'preferred_batch_size': [5, 6],
                'max_queue_delay_microseconds': 200
            },
            'instance_group': [{
                'kind': 'KIND_GPU',
                'count': 2
            }]
        }, {
            'dynamic_batching': {
                'preferred_batch_size': [5, 6],
                'max_queue_delay_microseconds': 200
            },
            'instance_group': [{
                'kind': 'KIND_CPU',
                'count': 1
            }]
        }, {
            'dynamic_batching': {
                'preferred_batch_size': [5, 6],
                'max_queue_delay_microseconds': 200
            },
            'instance_group': [{
                'kind': 'KIND_CPU',
                'count': 2
            }]
        }]
        self.assertEqual(expected_model_configs, model_configs)

    def test_generate_run_config_for_model_sweep(self):
        # remote launch mode, no model sweeps
        args = [
            'model-analyzer', 'profile', '--model-repository',
            'cli_repository', '-f', 'path-to-config-file',
            '--triton-launch-mode', 'remote'
        ]
        yaml_content = """
            concurrency: [1, 2, 3]
            batch_sizes: [2, 3, 4]
            profile_models:
            - vgg_16_graphdef
            """
        config = self._evaluate_config(args, yaml_content)
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        self.assertEqual(len(model_configs), 1)
        self.mock_client.set_model_config(
            {'config': {
                'name': 'vgg_16_graphdef'
            }})
        run_config_generator.generate_run_config_for_model_sweep(
            config.profile_models[0], model_configs[0])
        self.assertEqual(len(run_config_generator.run_configs()), 9)
        self.assertEqual(
            run_config_generator.run_configs()[0].model_config().get_field(
                'name'), 'vgg_16_graphdef')

        # remote mode, with model sweeps
        yaml_content = """
            concurrency: [1, 2, 3]
            batch_sizes: [2, 3, 4]
            profile_models:
            -
                vgg_16_graphdef:
                    model_config_parameters:
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1, 2]
            """

        config = self._evaluate_config(args, yaml_content)
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        self.assertEqual(len(model_configs), 2)
        self.mock_client.set_model_config(
            {'config': {
                'name': 'vgg_16_graphdef'
            }})
        for model_config in model_configs:
            run_config_generator.generate_run_config_for_model_sweep(
                config.profile_models[0], model_config)
        self.assertEqual(len(run_config_generator.run_configs()), 18)
        self.assertEqual(
            run_config_generator.run_configs()[0].model_config().get_field(
                'name'), 'vgg_16_graphdef')

        # Not remote, no model sweep
        args = [
            'model-analyzer', 'profile', '--model-repository',
            'cli_repository', '-f', 'path-to-config-file'
        ]
        yaml_content = """
            concurrency: [1, 2, 3]
            batch_sizes: [2, 3, 4]
            profile_models:
            - vgg_16_graphdef
            """

        config = self._evaluate_config(args, yaml_content)
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        self.assertEqual(len(model_configs), 1)
        run_config_generator.generate_run_config_for_model_sweep(
            config.profile_models[0], model_configs[0])
        self.assertEqual(len(run_config_generator.run_configs()), 9)
        self.assertEqual(
            run_config_generator.run_configs()[0].model_config().get_field(
                'name'), 'vgg_16_graphdef_i0')

        # Not remote, with model sweep
        args = [
            'model-analyzer', 'profile', '--model-repository',
            'cli_repository', '-f', 'path-to-config-file'
        ]
        yaml_content = """
            concurrency: [1, 2, 3]
            batch_sizes: [2, 3, 4]
            profile_models:
            -
                vgg_16_graphdef:
                    model_config_parameters:
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1, 2]
            """

        config = self._evaluate_config(args, yaml_content)
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        self.assertEqual(len(model_configs), 2)
        for model_config in model_configs:
            run_config_generator.generate_run_config_for_model_sweep(
                config.profile_models[0], model_config)
        self.assertEqual(len(run_config_generator.run_configs()), 18)
        self.assertEqual(
            run_config_generator.run_configs()[0].model_config().get_field(
                'name'), 'vgg_16_graphdef_i0')
        self.assertEqual(
            run_config_generator.run_configs()[9].model_config().get_field(
                'name'), 'vgg_16_graphdef_i1')

        # Test map fields
        yaml_content = """
            concurrency: [1, 2, 3]
            batch_sizes: [2, 3, 4]
            profile_models:
            -
                vgg_16_graphdef:
                    model_config_parameters:
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1, 2]
                        parameters:
                            MAX_SESSION_SHARE_COUNT: 
                              string_value: [1, 2, 3, 4, 5]
            """

        config = self._evaluate_config(args, yaml_content)
        run_config_generator = RunConfigGenerator(config=config,
                                                  client=self.client)
        model_configs = run_config_generator.generate_model_config_combinations(
            config.profile_models[0].model_config_parameters())
        self.assertEqual(len(model_configs), 10)
        self.mock_client.set_model_config(
            {'config': {
                'name': 'vgg_16_graphdef'
            }})
        for model_config in model_configs:
            run_config_generator.generate_run_config_for_model_sweep(
                config.profile_models[0], model_config)
        self.assertEqual(len(run_config_generator.run_configs()), 90)
        self.assertEqual(
            run_config_generator.run_configs()[0].model_config().get_field(
                'name'), 'vgg_16_graphdef_i0')

    def tearDown(self):
        self.mock_model_config.stop()
        self.mock_client.stop()
