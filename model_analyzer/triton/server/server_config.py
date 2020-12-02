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
