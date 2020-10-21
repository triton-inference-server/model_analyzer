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

from model_analyzer.triton.server.server_docker_factory import TritonServerDockerFactory
from model_analyzer.triton.client.http_client_factory import TritonHTTPClientFactory
from model_analyzer.triton.client.grpc_client_factory import TritonGRPCClientFactory
from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.client.client_config import TritonClientConfig
from model_analyzer.triton.model.model import Model
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

# Test parameters
MODEL_LOCAL_PATH = '/model_analyzer/models'
MODEL_REPOSITORY_PATH = '/model_analyzer/models'
TRITON_VERSION = '20.09'
CONFIG_TEST_ARG = 'url'
CLIENT_TEST_PROTOCOLS = [(TritonHTTPClientFactory(), 'localhost:8000'),
                         (TritonGRPCClientFactory(), 'localhost:8001')]
TEST_MODEL_NAME = 'classification_chestxray_v1'


class TestTritonClientMethods(unittest.TestCase):

    def setUp(self):
        # Create server config
        self.server_config = TritonServerConfig()
        self.server_config['model-repository'] = MODEL_REPOSITORY_PATH
        self.server_config['model-control-mode'] = 'explicit'

        # Set CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # Create and start the server
        factory = TritonServerDockerFactory()
        self.server = factory.create_server(
            model_path=MODEL_LOCAL_PATH,
            version=TRITON_VERSION,
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

        # Run for both types of protocols
        for factory, url in CLIENT_TEST_PROTOCOLS:
            client_config['url'] = url
            client = factory.create_client(config=client_config)

        # Try to create a client without specifying url and expect error
            with self.assertRaises(AssertionError,
                                   msg="Expected AssertionError for trying to "
                                   "create client without specifying url."):
                client_config['url'] = None
                client = factory.create_client(config=client_config)

    def test_wait_for_ready(self):
        # Create a TritonClientConfig
        client_config = TritonClientConfig()
        client_config['url'] = 'localhost:8000'

        # Create client
        factory = TritonHTTPClientFactory()
        client = factory.create_client(config=client_config)

        # Wait for the server when it hasnt been started
        expected_string = "Could not determine server readiness. Number of retries exceeded."
        failure_message = "Expected exception waiting for server which is not running"
        with self.assertRaisesRegex(TritonModelAnalyzerException,
                                    expected_regex=expected_string,
                                    msg=failure_message):
            client.wait_for_server_ready(num_retries=10)

        # Start the server
        self.server.start()

        # Wait for running server
        client.wait_for_server_ready()

        self.server.stop()

    def test_load_unload_model(self):
        # Create config
        client_config = TritonClientConfig()
        client_config['url'] = 'localhost:8001'

        # Create client
        factory = TritonGRPCClientFactory()
        client = factory.create_client(config=client_config)

        # Start the server and wait till it is ready
        self.server.start()
        client.wait_for_server_ready()

        # Try to load a dummy model and expect error
        with self.assertRaises(TritonModelAnalyzerException,
                               msg="Expected Exception trying"
                               " to load dummy model"):
            client.load_model(Model('dummy'))

        # Load the test model
        client.load_model(Model(TEST_MODEL_NAME))
        client.wait_for_model_ready(Model(TEST_MODEL_NAME), num_retries=10)

        # Try to unload a dummy model
        client.unload_model(Model('dummy'))

        # Unload the test model
        client.unload_model(Model(TEST_MODEL_NAME))

        self.server.stop()

    def tearDown(self):
        # In case test raises exception
        self.server.stop()
        pass


if __name__ == '__main__':
    unittest.main()
