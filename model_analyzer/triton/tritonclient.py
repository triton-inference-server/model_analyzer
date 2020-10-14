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

import time

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

WAIT_FOR_READY_NUM_RETRIES = 100

class TritonClientConfig:
    def __init__(self):
        self._client_args = {
            'url' : None
        }

    def __getitem__(self, key):
        return self._client_args[key]
    
    def __setitem__(self, key, value):
        if key in self._client_args:
            self._client_args[key] = value
        else:
            raise Exception("The argument '{}' to the Triton Inference Server"
                             " is not supported by the model analyzer.".format(key))

class TritonClientFactory:
    """
    A factory for creating Triton Client instances
    """
    def __init__(self):
        pass

    def create_client(self, protocol, config):
        if protocol is 'grpc':
            return TritonGRPCClient(config=config)
        elif protocol is 'http':
            return TritonHTTPClient(config=config)
        else:
            raise Exception("The protocol '{}' for Triton Inference Server"
                            " is not supported by the model analyzer.".format(protocol))

class TritonClient:
    """
    Defines the interface for the objects created by 
    TritonClientFactory
    """
    def __init__(self, config):
        self._client_config = config
        assert self._client_config['url'], \
            "Must specify url in client config."

    def wait_for_server_ready(self, num_retries=WAIT_FOR_READY_NUM_RETRIES):
        """
        Returns when server is ready
        """
        while num_retries > 0:
            try:
                if self._client.is_server_ready():
                    return
                else:
                    time.sleep(0.1)
                    num_retries -= 1
            except Exception as e:
                time.sleep(0.1)
                num_retries -= 1
                pass
        raise Exception("Could not determine server readiness. "
                        "Number of retries exceeded.")
    
    def load_model(self, model_name):
        """
        Request the inference server to load
        a particular model in explicit model
        control mode
        """
        try:
            self._client.load_model(model_name)
        except InferenceServerException as e:
            raise Exception("Unable to load the model : {}".format(e))

    def unload_model(self, model_name):
        """
        Request the inference server to unload
        a particular model in explicit model
        control mode
        """
        try:
            self._client.unload_model(model_name)
        except InferenceServerException as e:
            raise Exception("Unable to unload the model : {}".format(e))
    
    def wait_for_model_ready(self, model_name, num_retries=WAIT_FOR_READY_NUM_RETRIES):
        """
        Returns when model with name model_name
        is ready
        """
        while num_retries > 0:
            try:
                if self._client.is_model_ready(model_name):
                    return
                else:
                    time.sleep(0.05)
                    num_retries -= 1
            except Exception as e:
                time.sleep(0.05)
                num_retries -= 1
                pass
        raise Exception("Could not determine model readiness. "
                        "Number of retries exceeded.")
        
class TritonHTTPClient(TritonClient):
    """
    Concrete implementation of TritonClient
    for HTTP
    """
    def __init__(self, config):
        super().__init__(config=config)
        self._client = httpclient.InferenceServerClient(
                url=self._client_config['url'])

class TritonGRPCClient(TritonClient):
    """
    Concrete implementation of TritonClient
    for GRPC
    """
    def __init__(self, config):
        super().__init__(config=config)
        self._client = grpcclient.InferenceServerClient(
            url=self._client_config['url'])



