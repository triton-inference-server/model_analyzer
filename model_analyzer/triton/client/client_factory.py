# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .grpc_client import TritonGRPCClient
from .http_client import TritonHTTPClient


class TritonClientFactory:
    """
    Base client creator class that declares
    a factory method
    """

    @staticmethod
    def create_grpc_client(server_url, ssl_options={}):
        """
        Parameters
        ----------
        server_url : str
            The url for Triton server's GRPC endpoint
        ssl_options : dict
            Dictionary of SSL options for gRPC python client

        Returns
        -------
        TritonGRPCClient
        """
        return TritonGRPCClient(server_url=server_url, ssl_options=ssl_options)

    @staticmethod
    def create_http_client(server_url, ssl_options={}):
        """
        Parameters
        ----------
        server_url : str
            The url for Triton server's HTTP endpoint
        ssl_options : dict
            Dictionary of SSL options for HTTP python client

        Returns
        -------
        TritonHTTPClient
        """
        return TritonHTTPClient(server_url=server_url, ssl_options=ssl_options)
