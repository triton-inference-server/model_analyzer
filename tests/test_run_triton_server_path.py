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
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from model_analyzer.config.input.config_utils  import binary_path_validator

class TestRemoteTritonServerPath(trc.TestResultCollector):

    _bogus_path = '/path/to/nowhere'
    _good_path = 'model-analyzer'

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

    def test_remote_launch_mode(self):
        args = [
            'model-analyzer', 'profile', 
            '--triton-server-path', self._bogus_path,
            '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file', 
            '--triton-launch-mode', 'remote', 
        ]
        yaml_content = """
            concurrency: [1, 2, 3]
            batch_sizes: [2, 3, 4]
            profile_models:
            - vgg_16_graphdef
            """

        self._evaluate_config(args, yaml_content)

    def test_bogus_local_launch_mode(self):
        args = [
            'model-analyzer', 'profile', 
            '--triton-server-path', self._bogus_path,
            '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file', 
            '--triton-launch-mode', 'local', 
        ]
        yaml_content = """
            concurrency: [1, 2, 3]
            batch_sizes: [2, 3, 4]
            profile_models:
            - vgg_16_graphdef
            """

        try:
            # Opt-in validator
            config = self._evaluate_config(args, yaml_content)
            tsp = config.get_config()['triton_server_path']
            path = tsp.value()
            tsp.set_validator(binary_path_validator)
            tsp.set_value(path)        
            assert False, "local launch mode needs to have validated triton-server-path"
        except TritonModelAnalyzerException as e:
            if self._bogus_path in str(e):
                pass
            else:
                assert False, "expecting " + self._bogus_path + " in exception"


    def test_good_local_launch_mode(self):
        args = [
            'model-analyzer', 'profile', 
            '--triton-server-path', self._good_path,
            '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file', 
            '--triton-launch-mode', 'local', 
        ]
        yaml_content = """
            concurrency: [1, 2, 3]
            batch_sizes: [2, 3, 4]
            profile_models:
            - vgg_16_graphdef
            """

        # Opt-in validator
        config = self._evaluate_config(args, yaml_content)
        tsp = config.get_config()['triton_server_path']
        path = tsp.value()
        tsp.set_validator(binary_path_validator)
        tsp.set_value(path)        


    def test_docker_launch_mode(self):
        args = [
            'model-analyzer', 'profile', 
            '--triton-server-path', self._bogus_path,
            '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file', 
            '--triton-launch-mode', 'docker', 
        ]
        yaml_content = """
            concurrency: [1, 2, 3]
            batch_sizes: [2, 3, 4]
            profile_models:
            - vgg_16_graphdef
            """

        self._evaluate_config(args, yaml_content)


    def test_c_api_launch_mode(self):
        args = [
            'model-analyzer', 'profile', 
            '--triton-server-path', self._bogus_path,
            '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file', 
            '--triton-launch-mode', 'c_api', 
        ]
        yaml_content = """
            concurrency: [1, 2, 3]
            batch_sizes: [2, 3, 4]
            profile_models:
            - vgg_16_graphdef
            """

        self._evaluate_config(args, yaml_content)


    def tearDown(self):
        self.mock_model_config.stop()
        self.mock_client.stop()
