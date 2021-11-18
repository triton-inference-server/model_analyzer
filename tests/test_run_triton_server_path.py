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

class TestRemoteTritonServerPath(trc.TestResultCollector):

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

    def test_remote(self):
        args = [
            'model-analyzer', 'profile', 
            '--triton-server-path', '/path/to/nowhere',
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

    def test_local(self):
        args = [
            'model-analyzer', 'profile', 
            '--triton-server-path', '/path/to/nowhere',
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
            config = self._evaluate_config(args, yaml_content)
            assert False, "local launch mode needs to have validated triton-server-path"
        except TritonModelAnalyzerException:
            pass


    def test_docker(self):
        args = [
            'model-analyzer', 'profile', 
            '--triton-server-path', '/path/to/nowhere',
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

        try:
            config = self._evaluate_config(args, yaml_content)
            assert False, "docker launch mode needs to have validated triton-server-path"
        except TritonModelAnalyzerException:
            pass


    def test_c_api(self):
        args = [
            'model-analyzer', 'profile', 
            '--triton-server-path', '/path/to/nowhere',
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

        try:
            config = self._evaluate_config(args, yaml_content)
            assert False, "c_api launch mode needs to have validated triton-server-path"
        except TritonModelAnalyzerException:
            pass


    def tearDown(self):
        self.mock_model_config.stop()
        self.mock_client.stop()
