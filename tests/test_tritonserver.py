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

from model_analyzer.triton.tritonserver import TritonServerConfig, TritonServerFactory

# Test parameters
MODEL_LOCAL_PATH = '/model_analyzer/models'
MODEL_REPOSITORY_PATH = '/model_analyzer/models'
TRITON_VERSION = '20.09'
SERVER_RUN_ENVIRONMENTS = ['docker', 'local']
CONFIG_TEST_ARG = 'exit-on-error'
CLI_TO_STRING_TEST_ARGS = {
    'allow-grpc' : True,
    'min-supported-compute-capability' : 7.5,
    'metrics-port' : 8000,
    'model-repository' : MODEL_REPOSITORY_PATH,
}

class TestTritonServerMethods(unittest.TestCase):

    def setUp(self):
        
        # Create a factory
        self.server_factory = TritonServerFactory()
        self.server = None
        
    def test_server_config(self):

        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH
        
        # Check config initializations
        self.assertIsNone(server_config[CONFIG_TEST_ARG], 
                            msg="Server config had unexpected initial"
                                "value for {}".format(CONFIG_TEST_ARG))
        # Set value
        server_config[CONFIG_TEST_ARG] = True

        # Test get again
        self.assertTrue(server_config[CONFIG_TEST_ARG], 
                            msg="{} was not set".format(CONFIG_TEST_ARG))
        
        # Try to set an unsupported config argument, expect failure
        try:
            server_config['dummy'] = 1
            self.assertTrue(False, msg="Expected ValueError on trying to set"
                                        "unsupported argument in Triton server"
                                        "config")
        except Exception as e:
            pass
        
        # Reset test arg
        server_config[CONFIG_TEST_ARG] = None
        
        # Finally set a couple of args and then check the cli string
        for arg,value in CLI_TO_STRING_TEST_ARGS.items():
            server_config[arg] = value

        cli_string = server_config.to_cli_string()

        for argstring in cli_string.split():

            # Parse the created string
            arg, value = argstring.split('=')
            arg = arg[2:]

            # Make sure each parsed arg was in test dict
            self.assertIn(arg, CLI_TO_STRING_TEST_ARGS,
                        msg="CLI string contained unknown argument: {}".format(arg))

            # Make sure parsed value is the one from dict, check type too
            test_value = CLI_TO_STRING_TEST_ARGS[arg]
            self.assertEqual(test_value, type(test_value)(value),
                             msg="CLI string contained unknown value: {}".format(value))


    def test_create_server(self):
        
        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH

        # Run for both types of environments
        for run_type in SERVER_RUN_ENVIRONMENTS:
            try:    
                self.server = self.server_factory.create_server(
                            run_type=run_type,
                            model_path=MODEL_LOCAL_PATH, 
                            version=TRITON_VERSION,
                            config=server_config)

            except Exception as e:
                self.assertTrue(False, msg=e)

        # Run dummy environment and expect ValueError
        try:
            self.server = self.server_factory.create_server(
                            run_type='dummy',
                            model_path=MODEL_LOCAL_PATH,
                            version=TRITON_VERSION,
                            config=server_config)
            self.assertTrue(False, msg="Expected ValueError for dummy server run environment.")
        except Exception as e:
            pass

        # Try to create a server without specifying model repository and expect error
        try:
            self.server = self.server_factory.create_server(
                            run_type='docker',
                            model_path=MODEL_LOCAL_PATH,
                            version=TRITON_VERSION,
                            config=server_config)
            self.assertTrue(False, msg="Expected AssertionError for trying to create"
                                       "server without specifying model repository.")
        except Exception as e:
            pass

    def test_start_wait_stop_gpus(self):

        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH

        # Create a server to run in docker 
        for run_type in SERVER_RUN_ENVIRONMENTS:    
            try:
                self.server = self.server_factory.create_server(
                                run_type=run_type,
                                model_path=MODEL_LOCAL_PATH, 
                                version=TRITON_VERSION,
                                config=server_config)
                
            except Exception as e:
                self.assertTrue(False, msg=e)
            
            # Set CUDA_VISIBLE_DEVICES and start the server
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'

            # Now start the server
            self.server.start()
            self.server.wait_for_ready()
            self.server.stop()

    def tearDown(self):

        # In case test raises exception
        if self.server is not None:
            self.server.stop()

if __name__ == '__main__':
    unittest.main()
    