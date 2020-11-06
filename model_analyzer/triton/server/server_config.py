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

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class TritonServerConfig:
    """
    A config class to set arguments to the Triton Inference
    Server. An argument set to None will use the server default.
    """

    def __init__(self):
        # Args will be a dict with the string representation as key
        self._server_args = {
            # Logging
            'log-verbose': None,
            'log-info': None,
            'log-warning': None,
            'log-error': None,
            'id': None,
            # Model Repository
            'model-store': None,
            'model-repository': None,
            # Exit
            'exit-timeout-secs': None,
            'exit-on-error': None,
            # Strictness
            'strict-model-config': None,
            'strict-readiness': None,
            # API Servers
            'allow-http': None,
            'http-port': None,
            'http-thread-count': None,
            'allow-grpc': None,
            'grpc-port': None,
            'grpc-infer-allocation-pool-size': None,
            'grpc-use-ssl': None,
            'grpc-server-cert': None,
            'grpc-server-key': None,
            'grpc-root-cert': None,
            'allow-metrics': None,
            'allow-gpu-metrics': None,
            'metrics-port': None,
            # Tracing
            'trace-file': None,
            'trace-level': None,
            'trace-rate': None,
            # Model control
            'model-control-mode': None,
            'repository-poll-secs': None,
            'load-model': None,
            # Memory and GPU
            'pinned-memory-pool-byte-size': None,
            'cuda-memory-pool-byte-size': None,
            'min-supported-compute-capability': None,
            # Backend config
            'backend-directory': None,
            'backend-config': None,
            'allow-soft-placement': None,
            'gpu-memory-fraction': None,
            'tensorflow-version': None
        }

    def to_cli_string(self):
        """
        Utility function to convert a config into a
        string of arguments to the server with CLI.

        Returns
        -------
        str
            the command consisting of all set arguments to
            the tritonserver.
            e.g. '--model-repository=/models --log-verbose=True'
        """

        return ' '.join([
            f'--{key}={val}' for key, val in self._server_args.items() if val
        ])

    def __getitem__(self, key):
        """
        Gets an arguments value in config

        Parameters
        ----------
        key : str
            The name of the argument to the tritonserver

        Returns
        -------
            The value that the argument is set to in this config
        """

        return self._server_args[key]

    def __setitem__(self, key, value):
        """
        Sets an arguments value in config
        after checking if defined/supported.

        Parameters
        ----------
        key : str
            The name of the argument to the tritonserver
        value : (any)
            The value to which the argument is being set

        Raises
        ------
        TritonModelAnalyzerException
            If key is unsupported or undefined in the
            config class
        """

        if key in self._server_args:
            self._server_args[key] = value
        else:
            raise TritonModelAnalyzerException(
                f"The argument '{key}' to the Triton Inference "
                "Server is not supported by the model analyzer.")
