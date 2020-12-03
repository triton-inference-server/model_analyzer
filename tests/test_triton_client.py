# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os
import unittest
import sys
sys.path.append('../common')

from .mock_server_docker import MockServerDockerMethods
from .mock_client import MockTritonClientMethods

from model_analyzer.triton.server.server_factory import TritonServerFactory
from model_analyzer.triton.client.client_factory import TritonClientFactory
from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.model.model import Model
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
import test_result_collector as trc

# Test parameters
MODEL_REPOSITORY_PATH = 'test_repo'
TRITON_IMAGE = 'test_image'
CONFIG_TEST_ARG = 'url'
GRPC_URL = 'test_grpc_url'
HTTP_URL = 'test_http_url'
TEST_MODEL_NAME = 'test_model'


class TestTritonClientMethods(trc.TestResultCollector):
    def setUp(self):

        # GPUs
        gpus = ['all']

        # Mocks
        self.mock_server_docker = MockServerDockerMethods()
        self.tritonclient_mock = MockTritonClientMethods()

        # Create server config
        self.server_config = TritonServerConfig()
        self.server_config['model-repository'] = MODEL_REPOSITORY_PATH
        self.server_config['model-control-mode'] = 'explicit'

        # Set CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # Create and start the server
        self.server = TritonServerFactory.create_server_docker(
            image=TRITON_IMAGE,
            config=self.server_config,
            gpus=gpus)

    def test_create_client(self):

        # Create GRPC client
        client = TritonClientFactory.create_grpc_client(server_url=GRPC_URL)
        self.tritonclient_mock.assert_created_grpc_client_with_args(
            url=GRPC_URL)

        # Create HTTP client
        client = TritonClientFactory.create_http_client(server_url=HTTP_URL)
        self.tritonclient_mock.assert_created_http_client_with_args(
            url=HTTP_URL)

    def test_wait_for_server_ready(self):

        # For reuse
        def _test_with_client(self, client):
            with self.assertRaises(TritonModelAnalyzerException,
                                   msg="Expected Exception trying"
                                   " wait for server ready"):
                self.tritonclient_mock.raise_exception_on_wait_for_server_ready(
                )
                client.wait_for_server_ready(num_retries=1)
            self.tritonclient_mock.reset()

            with self.assertRaises(TritonModelAnalyzerException,
                                   msg="Expected Exception on"
                                   " server not ready"):
                self.tritonclient_mock.set_server_not_ready()
                client.wait_for_server_ready(num_retries=5)

            self.tritonclient_mock.reset()
            client.wait_for_server_ready(num_retries=1)

        # HTTP client
        client = TritonClientFactory.create_http_client(server_url=HTTP_URL)
        self.tritonclient_mock.assert_created_http_client_with_args(HTTP_URL)
        _test_with_client(self, client)
        self.tritonclient_mock.assert_http_client_waited_for_server_ready()

        # GRPC client
        client = TritonClientFactory.create_grpc_client(server_url=GRPC_URL)
        self.tritonclient_mock.assert_created_grpc_client_with_args(GRPC_URL)
        _test_with_client(self, client)
        self.tritonclient_mock.assert_grpc_client_waited_for_server_ready()

    def test_wait_for_model_ready(self):

        # For reuse
        def _test_with_client(self, client):
            with self.assertRaises(TritonModelAnalyzerException,
                                   msg="Expected Exception trying"
                                   " wait for server ready"):
                self.tritonclient_mock.raise_exception_on_wait_for_model_ready(
                )
                client.wait_for_model_ready(model=Model(TEST_MODEL_NAME),
                                            num_retries=1)
            self.tritonclient_mock.reset()

            with self.assertRaises(TritonModelAnalyzerException,
                                   msg="Expected Exception on"
                                   " server not ready"):
                self.tritonclient_mock.set_model_not_ready()
                client.wait_for_model_ready(model=Model(TEST_MODEL_NAME),
                                            num_retries=5)
            self.tritonclient_mock.reset()
            client.wait_for_model_ready(model=Model(TEST_MODEL_NAME),
                                        num_retries=1)

        # HTTP client
        client = TritonClientFactory.create_http_client(server_url=HTTP_URL)
        self.tritonclient_mock.assert_created_http_client_with_args(HTTP_URL)
        _test_with_client(self, client)
        self.tritonclient_mock.assert_http_client_waited_for_model_ready(
            model_name=TEST_MODEL_NAME)

        # GRPC client
        client = TritonClientFactory.create_grpc_client(server_url=GRPC_URL)
        self.tritonclient_mock.assert_created_grpc_client_with_args(GRPC_URL)
        _test_with_client(self, client)
        self.tritonclient_mock.assert_grpc_client_waited_for_model_ready(
            model_name=TEST_MODEL_NAME)

    def test_load_unload_model(self):

        # Create client
        client = TritonClientFactory.create_grpc_client(server_url=GRPC_URL)
        self.tritonclient_mock.assert_created_grpc_client_with_args(GRPC_URL)

        # Start the server and wait till it is ready
        self.server.start()
        client.wait_for_server_ready(num_retries=1)

        # Try to load a dummy model and expect error
        with self.assertRaises(TritonModelAnalyzerException,
                               msg="Expected Exception trying"
                               " to load dummy model"):
            self.tritonclient_mock.raise_exception_on_load()
            client.load_model(Model('dummy'))

        self.tritonclient_mock.reset()

        # Load the test model
        client.load_model(Model(TEST_MODEL_NAME))
        client.wait_for_model_ready(Model(TEST_MODEL_NAME), num_retries=1)

        # Try to unload a dummy model and expect error
        with self.assertRaises(TritonModelAnalyzerException,
                               msg="Expected Exception trying"
                               " to unload dummy model"):
            self.tritonclient_mock.raise_exception_on_unload()
            client.unload_model(Model('dummy'))

        self.tritonclient_mock.reset()

        # Unload the test model
        client.unload_model(Model(TEST_MODEL_NAME))

        self.server.stop()

    def tearDown(self):
        # In case test raises exception
        if self.server is not None:
            self.server.stop()
        self.mock_server_docker.stop()
        self.tritonclient_mock.stop()


if __name__ == '__main__':
    unittest.main()
