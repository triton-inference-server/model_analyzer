# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class TritonServerConfig:
    """
    A config class to set arguments to the Triton Inference
    Server. An argument set to None will use the server default.
    """
    server_arg_keys = [
        # Logging
        'log-verbose',
        'log-info',
        'log-warning',
        'log-error',
        'id',
        # Model Repository
        'model-store',
        'model-repository',
        # Exit
        'exit-timeout-secs',
        'exit-on-error',
        # Strictness
        'strict-model-config',
        'strict-readiness',
        # API Servers
        'allow-http',
        'http-port',
        'http-thread-count',
        'allow-grpc',
        'grpc-port',
        'grpc-infer-allocation-pool-size',
        'grpc-use-ssl',
        'grpc-server-cert',
        'grpc-server-key',
        'grpc-root-cert',
        'allow-metrics',
        'allow-gpu-metrics',
        'metrics-port',
        # Tracing
        'trace-file',
        'trace-level',
        'trace-rate',
        # Model control
        'model-control-mode',
        'repository-poll-secs',
        'load-model',
        # Memory and GPU
        'pinned-memory-pool-byte-size',
        'cuda-memory-pool-byte-size',
        'min-supported-compute-capability',
        # Backend config
        'backend-directory',
        'backend-config',
        'allow-soft-placement',
        'gpu-memory-fraction',
        'tensorflow-version'
    ]

    def __init__(self):
        """
        Construct TritonServerConfig
        """

        self._server_args = {k: None for k in self.server_arg_keys}

    @classmethod
    def allowed_keys(cls):
        """
        Returns
        -------
        list of str
            The keys that can be used to configure tritonserver instance
        """

        snake_cased_keys = [
            key.replace('-', '_') for key in cls.server_arg_keys
        ]
        return cls.server_arg_keys + snake_cased_keys

    def update_config(self, params=None):
        """
        Allows setting values from a
        params dict

        Parameters
        ----------
        params: dict
            keys are allowed args to perf_analyzer
        """

        if params:
            for key in params:
                self[key.strip().replace('_', '-')] = params[key]

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

        return ' '.join(
            [f'--{key}={val}' for key, val in self._server_args.items() if val])

    def copy(self):
        """
        Returns
        -------
        TritonServerConfig
            object that has the same args as this one
        """

        config_copy = TritonServerConfig()
        config_copy.update_config(params=self._server_args)
        return config_copy

    def server_args(self):
        """
        Returns
        -------
        dict
            keys are server arguments
            values are their values
        """

        return self._server_args

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

        return self._server_args[key.strip().replace('_', '-')]

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

        kebab_cased_key = key.strip().replace('_', '-')
        if kebab_cased_key in self._server_args:
            self._server_args[kebab_cased_key] = value
        else:
            raise TritonModelAnalyzerException(
                f"The argument '{key}' to the Triton Inference "
                "Server is not supported by the model analyzer.")
