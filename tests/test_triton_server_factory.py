# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
from .common import test_result_collector as trc

from model_analyzer.triton.server.server_factory import TritonServerFactory
from unittest.mock import patch, MagicMock
from .mocks.mock_config import MockConfig
from .mocks.mock_os import MockOSMethods

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.cli.cli import CLI


class TestTritonServerFactory(trc.TestResultCollector):

    def setUp(self):
        # Mock path validation
        self.mock_os = MockOSMethods(
            mock_paths=['model_analyzer.triton.server.server_factory'])
        self.mock_os.start()

    def test_get_server_handle_remote(self):
        self._test_get_server_handle_helper(launch_mode="remote",
                                            use_dcgm=False,
                                            expected_local=1,
                                            expected_docker=0,
                                            expect_config=False)
        self._test_get_server_handle_helper(launch_mode="remote",
                                            use_dcgm=True,
                                            expected_local=1,
                                            expected_docker=0,
                                            expect_config=False)

    def test_get_server_handle_c_api(self):
        self._test_get_server_handle_helper(launch_mode="c_api",
                                            use_dcgm=False,
                                            expected_local=1,
                                            expected_docker=0,
                                            expect_config=False)
        self._test_get_server_handle_helper(launch_mode="c_api",
                                            use_dcgm=True,
                                            expected_local=1,
                                            expected_docker=0,
                                            expect_config=False)

    def test_get_server_handle_local(self):
        self._test_get_server_handle_helper(launch_mode="local",
                                            use_dcgm=False,
                                            expected_local=1,
                                            expected_docker=0,
                                            expect_config=True)
        self._test_get_server_handle_helper(launch_mode="local",
                                            use_dcgm=True,
                                            expected_local=1,
                                            expected_docker=0,
                                            expect_config=True)

    def test_get_server_handle_docker(self):
        self._test_get_server_handle_helper(launch_mode="docker",
                                            use_dcgm=False,
                                            expected_local=0,
                                            expected_docker=1,
                                            expect_config=True)
        self._test_get_server_handle_helper(launch_mode="docker",
                                            use_dcgm=True,
                                            expected_local=0,
                                            expected_docker=1,
                                            expect_config=True)

    def _test_get_server_handle_helper(self, launch_mode, expected_local,
                                       expected_docker, expect_config,
                                       use_dcgm):
        config = ConfigCommandProfile()
        config.triton_launch_mode = launch_mode
        config.triton_http_endpoint = "1:2"
        config.triton_grpc_endpoint = "3:4"
        config.monitoring_interval = 0.5
        config.use_local_gpu_monitor = use_dcgm

        with patch('model_analyzer.triton.server.server_factory.TritonServerFactory.create_server_local') as mocked_local, \
             patch('model_analyzer.triton.server.server_factory.TritonServerFactory.create_server_docker') as mocked_docker, \
             patch('model_analyzer.triton.server.server_factory.TritonServerFactory._validate_triton_install_path'), \
             patch('model_analyzer.triton.server.server_factory.TritonServerFactory._validate_triton_server_path'):
            _ = TritonServerFactory.get_server_handle(config, MagicMock(),
                                                      False)
            self.assertEqual(mocked_local.call_count, expected_local)
            self.assertEqual(mocked_docker.call_count, expected_docker)

            if expected_local:
                args, kwargs = mocked_local.call_args
            else:
                args, kwargs = mocked_docker.call_args
            triton_config = kwargs['config']

            if expect_config:
                self.assertEqual(triton_config['http-port'], '2')
                self.assertEqual(triton_config['grpc-port'], '4')
                if not use_dcgm:
                    self.assertEqual(triton_config['metrics-interval-ms'], 500)
            else:
                self.assertEqual(triton_config['http-port'], None)
                self.assertEqual(triton_config['grpc-port'], None)
                self.assertEqual(triton_config['metrics-interval-ms'], None)

    def tearDown(self):
        patch.stopall()

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


if __name__ == '__main__':
    unittest.main()
