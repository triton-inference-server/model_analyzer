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

import gevent.ssl
import logging

from .client import TritonClient
import tritonclient.http as httpclient


class TritonHTTPClient(TritonClient):
    """
    Concrete implementation of TritonClient
    for HTTP
    """

    def __init__(self, server_url, ssl_options={}):
        """
        Parameters
        ----------
        server_url : str
            The url for Triton server's HTTP endpoint
        ssl_options : dict
            Dictionary of SSL options for HTTP python client
        """

        ssl = False
        client_ssl_options = {}
        ssl_context_factory = gevent.ssl._create_unverified_context
        insecure = True
        verify_peer = 0
        verify_host = 0

        if server_url.startswith('http://'):
            server_url = server_url.replace('http://', '', 1)
        elif server_url.startswith('https://'):
            ssl = True
            server_url = server_url.replace('https://', '', 1)
        if 'ssl-https-ca-certificates-file' in ssl_options:
            client_ssl_options['ca_certs'] = ssl_options[
                'ssl-https-ca-certificates-file']
        if 'ssl-https-client-certificate-file' in ssl_options:
            if 'ssl-https-client-certificate-type' in ssl_options and ssl_options[
                    'ssl-https-client-certificate-type'] == 'PEM':
                client_ssl_options['certfile'] = ssl_options[
                    'ssl-https-client-certificate-file']
            else:
                logging.warning(
                    'model-analyzer with SSL must be passed a client certificate file in PEM format.'
                )
        if 'ssl-https-private-key-file' in ssl_options:
            if 'ssl-https-private-key-type' in ssl_options and ssl_options[
                    'ssl-https-private-key-type'] == 'PEM':
                client_ssl_options['keyfile'] = ssl_options[
                    'ssl-https-private-key-file']
            else:
                logging.warning(
                    'model-analyzer with SSL must be passed a private key file in PEM format.'
                )
        if 'ssl-https-verify-peer' in ssl_options:
            verify_peer = ssl_options['ssl-https-verify-peer']
        if 'ssl-https-verify-host' in ssl_options:
            verify_host = ssl_options['ssl-https-verify-host']
        if verify_peer != 0 and verify_host != 0:
            ssl_context_factory = None
            insecure = False

        self._client = httpclient.InferenceServerClient(
            url=server_url,
            ssl=ssl,
            ssl_options=client_ssl_options,
            ssl_context_factory=ssl_context_factory,
            insecure=insecure)
