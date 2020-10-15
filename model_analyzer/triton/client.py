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
        """Create function that dispatches the 
            correct server implementation based 
            on the protocol

        Parameters
        ----------
        protocol: str
            The protocol that the client uses to 
            communicate with the server
        
        config: TritonClientCnfig
        """
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
