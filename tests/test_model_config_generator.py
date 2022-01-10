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

from model_analyzer.config.generate.model_config_generator import ModelConfigGenerator
from model_analyzer.config.input.config_command_profile \
     import ConfigCommandProfile
from model_analyzer.cli.cli import CLI
from .common import test_result_collector as trc
from .common.test_utils import convert_to_bytes
from .mocks.mock_config import MockConfig
from .mocks.mock_os import MockOSMethods


class TestModelConfigGenerator(trc.TestResultCollector):

    def test_local_no_params(self):
        ''' 
        Test local mode with no model_config_parameters specified
        
        It will just sweep instance count (with dynamic batching on), and 
        default config (None) will be included 
        '''

        # yapf: disable
        yaml_content = convert_to_bytes("""
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'dynamic_batching': {}, 'instance_count': 1},
            {'dynamic_batching': {}, 'instance_count': 2},
            {'dynamic_batching': {}, 'instance_count': 3},
            {'dynamic_batching': {}, 'instance_count': 4},
            {'dynamic_batching': {}, 'instance_count': 5},
            None
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_content,
                                                  expected_configs)

    def test_local_no_params_search_disable(self):
        ''' 
        Test local mode with no model_config_parameters specified and run_search disabled
        
        This will just return empty config, since there are no parameters to combine
        '''

        # yapf: disable
        yaml_content = convert_to_bytes("""
            run_config_search_disable: True
            profile_models:
                - my-model
            """)

        expected_configs = [None]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_content,
                                                  expected_configs)

    def test_local_yes_params_search_disable(self):
        ''' 
        Test local mode with model_config_parameters specified and run_search disabled
        
        This will just combine all model_config_parameters
        '''

        # yapf: disable
        yaml_content = convert_to_bytes("""
            run_config_search_max_instance_count: 16
            run_config_search_disable: True
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,4,16]
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1,2]                        
            """)

        expected_configs = [
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 1},
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 4},
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 16},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 1},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 4},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 16},
            None
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_content,
                                                  expected_configs)

    def test_local_max_instance_count(self):
        ''' 
        Test that ModelConfigGenerator will honor run_config_search_max_instance_count
        '''

        # yapf: disable
        yaml_content = convert_to_bytes("""
            run_config_search_max_instance_count: 3
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'dynamic_batching': {}, 'instance_count': 1},
            {'dynamic_batching': {}, 'instance_count': 2},
            {'dynamic_batching': {}, 'instance_count': 3},
            None
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_content,
                                                  expected_configs)

    def test_local_yes_params_specified(self):
        ''' 
        Test local mode with model_config_parameters specified
        
        It will combine all legal combinations of config values, and 
        default config (None) will be included 
        '''

        # yapf: disable
        yaml_content = convert_to_bytes("""
            run_config_search_max_instance_count: 16
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,4,16]
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1,2]                        
            """)

        expected_configs = [
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 1},
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 4},
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 16},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 1},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 4},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 16},
            None
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_content,
                                                  expected_configs)

    def test_remote_yes_params_specified(self):
        ''' 
        Test remote mode with model_config_parameters specified
        
        It should always return a single empty config in remote mode
        '''

        # yapf: disable
        yaml_content = convert_to_bytes("""
            triton_launch_mode: remote            
            run_config_search_max_instance_count: 16
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,4,16]
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1,2]                        
            """)

        expected_configs = [None]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_content,
                                                  expected_configs)

    def test_remote_no_params_specified(self):
        ''' 
        Test remote mode with no model_config_parameters specified
        
        It should always return a single empty config in remote mode
        '''

        # yapf: disable
        yaml_content = convert_to_bytes("""
            triton_launch_mode: remote            
            run_config_search_max_instance_count: 16
            profile_models:
                - my-model
            """)

        expected_configs = [None]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_content,
                                                  expected_configs)

    def _run_and_test_model_config_generator(self, yaml_content,
                                             expected_configs):
        args = [
            'model-analyzer', 'profile', '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file'
        ]

        config = self._evaluate_config(args, yaml_content)
        mcg = ModelConfigGenerator(config, config.profile_models[0])
        model_configs = []
        while not mcg.is_done():
            model_configs.append(mcg.next_config())

        self.assertEqual(len(expected_configs), len(model_configs))
        for config in expected_configs:
            self.assertIn(config, model_configs)

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
