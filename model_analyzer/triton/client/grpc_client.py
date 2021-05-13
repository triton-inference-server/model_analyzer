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

from .client import TritonClient
import tritonclient.grpc as grpcclient


class TritonGRPCClient(TritonClient):
    """
    Concrete implementation of TritonClient
    for GRPC
    """

    def __init__(self, server_url):
        """
        Parameters
        ----------
        server_url : str
            The url for Triton server's GRPC endpoint
        """

        self._client = grpcclient.InferenceServerClient(url=server_url)

    def get_model_config(self, model_name, num_retries):
        """
        Model name to get the config for.

        Parameters
        ----------
        model_name : str
            Name of the model to find the config.

        num_retries : int
            Number of times to wait for the model load

        Returns
        -------
        dict
            A dictionary containg the model config.
        """

        self.load_model(model_name)
        self.wait_for_model_ready(model_name, num_retries)
        model_config_dict = self._client.get_model_config(model_name,
                                                          as_json=True)
        self.unload_model(model_name)
        return model_config_dict['config']
