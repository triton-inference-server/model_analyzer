# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import unittest
import sys
sys.path.append('../common')
from unittest.mock import patch, MagicMock

from .mock_server_docker import MockServerDockerMethods
from .mock_client import MockTritonClientMethods

from model_analyzer.triton.server.server_factory import TritonServerFactory
from model_analyzer.triton.client.client_factory import TritonClientFactory
from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.client.client_config import TritonClientConfig
from model_analyzer.triton.model.model import Model
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
import test_result_collector as trc

# Test parameters
MODEL_LOCAL_PATH = 'test_local_path'
MODEL_REPOSITORY_PATH = 'test_repo'
TRITON_IMAGE = 'test_image'
CONFIG_TEST_ARG = 'url'
GRPC_URL = 'test_grpc_url'
HTTP_URL = 'test_http_url'
TEST_MODEL_NAME = 'test_model'


class TestTritonClientMethods(trc.TestResultCollector):

    def setUp(self):
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
            model_path=MODEL_LOCAL_PATH,
            image=TRITON_IMAGE,
            config=self.server_config)

    def test_client_config(self):
        # Create a TritonClientConfig
        client_config = TritonClientConfig()

        # Check config initializations
        self.assertIsNone(client_config[CONFIG_TEST_ARG],
                          msg="Client config had unexpected initial "
                          f"value for {CONFIG_TEST_ARG}")
        # Set value
        client_config[CONFIG_TEST_ARG] = 'localhost:8001'

        # Test get again
        self.assertTrue(client_config[CONFIG_TEST_ARG],
                        msg=f"{CONFIG_TEST_ARG} was not set")

        # Try to set an unsupported config argument, expect failure
        with self.assertRaises(TritonModelAnalyzerException,
                               msg="Expected ValueError on trying to set "
                               "unsupported argument in Triton server "
                               "config"):
            client_config['dummy'] = 1

    def test_create_client(self):
        # Create a TritonServerConfig
        client_config = TritonClientConfig()

        # Create GRPC client
        client_config['url'] = GRPC_URL
        client = TritonClientFactory.create_grpc_client(config=client_config)

        # Try to create a client without specifying url and expect error
        with self.assertRaises(AssertionError,
                               msg="Expected AssertionError for trying to "
                               "create client without specifying url."):
            client_config['url'] = None
            client = TritonClientFactory.create_grpc_client(
                config=client_config)

        # Create HTTP client
        client_config['url'] = HTTP_URL
        client = TritonClientFactory.create_http_client(config=client_config)

        # Try to create a client without specifying url and expect error
        with self.assertRaises(AssertionError,
                               msg="Expected AssertionError for trying to "
                               "create client without specifying url."):
            client_config['url'] = None
            client = TritonClientFactory.create_http_client(
                config=client_config)

    @patch('model_analyzer.triton.server.server.requests', get=MagicMock())
    def test_load_unload_model(self, requests_mock):
        # Create config
        client_config = TritonClientConfig()
        client_config['url'] = GRPC_URL

        # Create client
        client = TritonClientFactory.create_grpc_client(config=client_config)

        # Start the server and wait till it is ready
        self.server.start()

        requests_mock.get.return_value.status_code = 200
        self.server.wait_for_ready(num_retries=1)

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
