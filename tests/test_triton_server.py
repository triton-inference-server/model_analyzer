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

from model_analyzer.triton.server import TritonServerConfig, TritonServerFactory

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
    'model-repository' : MODEL_REPOSITORY_PATH
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
            self.fail("Expected exception on trying to set"
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
                self.fail(e)

        # Run dummy environment and expect exception
        try:
            self.server = self.server_factory.create_server(
                            run_type='dummy',
                            model_path=MODEL_LOCAL_PATH,
                            version=TRITON_VERSION,
                            config=server_config)
            self.fail("Expected exception for dummy server run environment.")
        except Exception as e:
            pass

        # Try to create a server without specifying model repository and expect error
        try:
            self.server = self.server_factory.create_server(
                            run_type='docker',
                            model_path=MODEL_LOCAL_PATH,
                            version=TRITON_VERSION,
                            config=server_config)
            self.fail("Expected AssertionError for trying to create"
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
                self.fail(e)
            
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
