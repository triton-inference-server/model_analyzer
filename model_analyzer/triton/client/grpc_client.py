#!/usr/bin/env python3

# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tritonclient.grpc as grpcclient

from .client import TritonClient


class TritonGRPCClient(TritonClient):
    """
    Concrete implementation of TritonClient
    for GRPC
    """

    def __init__(self, server_url, ssl_options={}):
        """
        Parameters
        ----------
        server_url : str
            The url for Triton server's GRPC endpoint
        ssl_options : dict
            Dictionary of SSL options for gRPC python client
        """

        ssl = False
        root_certificates = None
        private_key = None
        certificate_chain = None

        if "ssl-grpc-use-ssl" in ssl_options:
            ssl = ssl_options["ssl-grpc-use-ssl"].lower() == "true"
        if "ssl-grpc-root-certifications-file" in ssl_options:
            root_certificates = ssl_options["ssl-grpc-root-certifications-file"]
        if "ssl-grpc-private-key-file" in ssl_options:
            private_key = ssl_options["ssl-grpc-private-key-file"]
        if "ssl-grpc-certificate-chain-file" in ssl_options:
            certificate_chain = ssl_options["ssl-grpc-certificate-chain-file"]

        # Fix for gRPC 1.60.0+: Force IPv4 resolution for localhost connections
        # gRPC 1.60.0+ prefers IPv6, causing "localhost" to resolve to [::1]
        # On systems where IPv6 is not properly configured, this causes connection failures
        # Force IPv4 by using 127.0.0.1, which is more reliable across environments
        channel_args = None
        if "localhost" in server_url:
            server_url = server_url.replace("localhost", "127.0.0.1")
            # For SSL connections, override target name to match certificate
            if ssl:
                channel_args = [("grpc.ssl_target_name_override", "localhost")]

        self._client = grpcclient.InferenceServerClient(
            url=server_url,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain,
            channel_args=channel_args,
        )

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
            A dictionary containing the model config.
        """

        self.wait_for_model_ready(model_name, num_retries)
        model_config_dict = self._client.get_model_config(model_name, as_json=True)
        return model_config_dict["config"]

    def get_model_repository_index(self):
        """
        Returns the JSON dict holding the model repository index.
        """
        return self._client.get_model_repository_index(as_json=True)["models"]

    def is_model_ready(self, model_name: str) -> bool:
        """
        Returns true if the model is loaded on the server
        """
        return self._client.is_model_ready(model_name)
