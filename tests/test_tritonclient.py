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

from model_analyzer.triton.tritonserver import TritonServerFactory, TritonServerConfig
from model_analyzer.triton.tritonclient import TritonClientFactory, TritonClientConfig

# Test parameters
MODEL_LOCAL_PATH = '/model_analyzer/models'
MODEL_REPOSITORY_PATH = '/model_analyzer/models'
TRITON_VERSION = '20.09'
CONFIG_TEST_ARG = 'url'
CLIENT_TEST_PROTOCOLS = [('http', 'localhost:8000'),('grpc', 'localhost:8001')]
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
        self.server_factory = TritonServerFactory()
        try:    
            self.server = self.server_factory.create_server(
                        run_type='docker',
                        model_path=MODEL_LOCAL_PATH, 
                        version=TRITON_VERSION,
                        config=self.server_config)
        except Exception as e:
            self.fail(e)

        # Create client factory
        self.client_factory = TritonClientFactory()

    def test_client_config(self):

        # Create a TritonClientConfig
        client_config = TritonClientConfig()
        
        # Check config initializations
        self.assertIsNone(client_config[CONFIG_TEST_ARG], 
                            msg="Client config had unexpected initial "
                                "value for {}".format(CONFIG_TEST_ARG))
        # Set value
        client_config[CONFIG_TEST_ARG] = 'localhost:8001'

        # Test get again
        self.assertTrue(client_config[CONFIG_TEST_ARG], 
                            msg="{} was not set".format(CONFIG_TEST_ARG))
        
        # Try to set an unsupported config argument, expect failure
        try:
            client_config['dummy'] = 1
            self.fail(msg="Expected ValueError on trying to set "
                          "unsupported argument in Triton server "
                          "config")
        except Exception as e:
            pass
        
    def test_create_client(self):

        # Create a TritonServerConfig
        client_config = TritonClientConfig()
        
        # Run for both types of protocols
        for protocol, url in CLIENT_TEST_PROTOCOLS:
            client_config['url'] = url
            try:    
                client = self.client_factory.create_client(
                            protocol=protocol,
                            config=client_config)

            except Exception as e:
                self.fail(e)

        # Run dummy environment and expect ValueError
        try:
            client = self.client_factory.create_client(
                            protocol='dummy',
                            config=client_config)
            self.fail(msg="Expected ValueError for dummy client protocol.")
        except Exception as e:
            pass

        # Try to create a client without specifying url and expect error
        try:
            client_config['url'] = None
            client = self.client_factory.create_client(
                            protocol='grpc',
                            config=client_config)
            self.fail(msg="Expected AssertionError for trying to "
                          "create client without specifying url.")
        except Exception as e:
            pass
    
    def test_wait_for_ready(self):
        
        # Create a TritonClientConfig
        client_config = TritonClientConfig()

        # Create client
        client_config['url'] = 'localhost:8000'
        try:    
            client = self.client_factory.create_client(
                        protocol='http',
                        config=client_config)

        except Exception as e:
            self.fail(e)
        
        # Wait for the server when it hasnt been started
        try:
            client.wait_for_server_ready(num_retries=10)
            self.fail("Expected exception waiting for "
                       "server which is not running")
        except Exception as e:
            self.assertEqual(str(e), "Could not determine server readiness. "
                                     "Number of retries exceeded.")
        
        # Start the server
        self.server.start()

        # Wait for running server
        try:
            client.wait_for_server_ready()
        except Exception as e:
            self.fail(e)
        
        # Stop the server
        self.server.stop()
    
    def test_load_unload_model(self):
        
        # Create config
        client_config = TritonClientConfig()

        # Create client
        client_config['url'] = 'localhost:8001'
        try:    
            client = self.client_factory.create_client(
                        protocol='grpc',
                        config=client_config)

        except Exception as e:
            self.fail(e)
        
        # Start the server and wait till it is ready
        try:
            self.server.start()
            client.wait_for_server_ready()
        except Exception as e:
            self.fail(e)

        # Try to load a dummy model and expect error
        try:
            client.load_model('dummy')
            self.fail("Expected Exception on trying to load dummy model")
        except Exception as e:
            pass
        
        # Load the test model
        try:
            client.load_model(TEST_MODEL_NAME)
            client.wait_for_model_ready(TEST_MODEL_NAME, num_retries=10)
        except Exception as e:
            self.fail(e)

        # Try to unload a dummy model and expect error
        try:
            client.unload_model('dummy')
            self.fail("Expected Exception on trying to unload dummy model")
        except Exception as e:
            pass

        # Unload the test model
        try:    
            client.unload_model(TEST_MODEL_NAME)
        except Exception as e:
            self.fail(e)
        
        # Stop the server
        self.server.stop()

    def tearDown(self):

        # In case test raises exception 
        self.server.stop()
        pass

if __name__ == '__main__':
    unittest.main()
    