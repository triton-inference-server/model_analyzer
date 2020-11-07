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

import requests
import time
from abc import ABC, abstractmethod

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

WAIT_FOR_READY_NUM_RETRIES = 100
SERVER_HTTP_PORT = 8000


class TritonServer(ABC):
    """
    Defines the interface for the objects created by
    TritonServerFactory
    """

    @abstractmethod
    def start(self):
        """
        Starts the tritonserver
        """

    @abstractmethod
    def stop(self):
        """
        Stops and cleans up after the server
        """

    def wait_for_ready(self, num_retries=WAIT_FOR_READY_NUM_RETRIES):
        """
        Parameters
        ----------
        num_retries : int
            number of times to send a ready status
            request to the server before raising
            an exception

        Raises
        ------
        TritonModelAnalyzerException
            1)  If config doesn't allow http
                requests
            2)  If server readiness could not be
                determined in given num_retries.
        """

        if self._server_config['allow-http'] is not False:
            http_port = self._server_config['http-port'] or SERVER_HTTP_PORT
            url = f"http://localhost:{http_port}/v2/health/ready"
        else:
            # TODO to use GRPC to check for ready also
            raise TritonModelAnalyzerException(
                'allow-http must be True in order to use wait_for_server_ready'
            )

        retries = num_retries

        # poll ready endpoint for number of retries
        while retries > 0:
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    return True
            except requests.exceptions.RequestException as e:
                pass
            time.sleep(0.1)
            retries -= 1

        # If num_retries is exceeded return an exception
        raise TritonModelAnalyzerException(
            f"Server not ready : num_retries : {num_retries}")
