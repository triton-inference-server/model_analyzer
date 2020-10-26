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

from model_analyzer.triton.server.server_factory import TritonServerFactory
from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

# Test parameters
MODEL_LOCAL_PATH = '/model_analyzer/models'
MODEL_REPOSITORY_PATH = '/model_analyzer/models'
TRITON_VERSION = '20.09'
CONFIG_TEST_ARG = 'exit-on-error'
CLI_TO_STRING_TEST_ARGS = {
    'allow-grpc': True,
    'min-supported-compute-capability': 7.5,
    'metrics-port': 8000,
    'model-repository': MODEL_REPOSITORY_PATH
}


class TestTritonServerMethods(unittest.TestCase):
    def setUp(self):

        # server setup
        self.server = None

    def test_server_config(self):

        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH

        # Check config initializations
        self.assertIsNone(server_config[CONFIG_TEST_ARG],
                          msg="Server config had unexpected initial"
                          f"value for {CONFIG_TEST_ARG}")
        # Set value
        server_config[CONFIG_TEST_ARG] = True

        # Test get again
        self.assertTrue(server_config[CONFIG_TEST_ARG],
                        msg=f"{CONFIG_TEST_ARG} was not set")

        # Try to set an unsupported config argument, expect failure
        with self.assertRaises(TritonModelAnalyzerException,
                               msg="Expected exception on trying to set"
                               "unsupported argument in Triton server"
                               "config"):
            server_config['dummy'] = 1

        # Reset test arg
        server_config[CONFIG_TEST_ARG] = None

        # Finally set a couple of args and then check the cli string
        for arg, value in CLI_TO_STRING_TEST_ARGS.items():
            server_config[arg] = value

        cli_string = server_config.to_cli_string()

        for argstring in cli_string.split():

            # Parse the created string
            arg, value = argstring.split('=')
            arg = arg[2:]

            # Make sure each parsed arg was in test dict
            self.assertIn(arg,
                          CLI_TO_STRING_TEST_ARGS,
                          msg=f"CLI string contained unknown argument: {arg}")

            # Make sure parsed value is the one from dict, check type too
            test_value = CLI_TO_STRING_TEST_ARGS[arg]
            self.assertEqual(
                test_value,
                type(test_value)(value),
                msg=f"CLI string contained unknown value: {value}")

    def test_create_server(self):

        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH

        # Run for both types of environments
        self.server = TritonServerFactory.create_server_docker(
            model_path=MODEL_LOCAL_PATH,
            version=TRITON_VERSION,
            config=server_config)

        self.server = TritonServerFactory.create_server_local(
            version=TRITON_VERSION, config=server_config)

        # Try to create a server without specifying model repository and expect
        # error
        server_config['model-repository'] = None
        with self.assertRaises(
                AssertionError,
                msg="Expected AssertionError for trying to create"
                "server without specifying model repository."):
            self.server = TritonServerFactory.create_server_docker(
                model_path=MODEL_LOCAL_PATH,
                version=TRITON_VERSION,
                config=server_config)
        with self.assertRaises(
                AssertionError,
                msg="Expected AssertionError for trying to create"
                "server without specifying model repository."):
            self.server = TritonServerFactory.create_server_local(
                version=TRITON_VERSION, config=server_config)

    def test_start_wait_stop_gpus(self):

        # Create a TritonServerConfig
        server_config = TritonServerConfig()
        server_config['model-repository'] = MODEL_REPOSITORY_PATH
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # Create server in docker, start , wait, and stop
        self.server = TritonServerFactory.create_server_docker(
            model_path=MODEL_LOCAL_PATH,
            version=TRITON_VERSION,
            config=server_config)

        # Set CUDA_VISIBLE_DEVICES and start the server
        with self.assertRaises(TritonModelAnalyzerException,
                               msg="Expected to exceed num_retries"):
            self.server.wait_for_ready()

        self.server.start()
        self.server.wait_for_ready()
        self.server.stop()

        # Create server locally, start , wait, and stop
        self.server = TritonServerFactory.create_server_local(
            version=TRITON_VERSION, config=server_config)

        with self.assertRaises(TritonModelAnalyzerException,
                               msg="Expected to exceed num_retries"):
            self.server.wait_for_ready()

        self.server.start()
        self.server.wait_for_ready()
        self.server.stop()

    def tearDown(self):

        # In case test raises exception
        if self.server is not None:
            self.server.stop()


if __name__ == '__main__':
    unittest.main()
