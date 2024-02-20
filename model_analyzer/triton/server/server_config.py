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

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class TritonServerConfig:
    """
    A config class to set arguments to the Triton Inference
    Server. An argument set to None will use the server default.
    """

    server_arg_keys = [
        # Server
        "id",
        "exit-timeout-secs",
        # Logging
        "log-verbose",
        "log-info",
        "log-warning",
        "log-error",
        "log-format",
        "log-file",
        # Model Repository
        "model-store",
        "model-repository",
        "exit-on-error",
        "disable-auto-complete-config",
        "strict-readiness",
        "model-control-mode",
        "repository-poll-secs",
        "load-model",
        "model-load-thread-count",
        "model-load-retry-count",
        "model-namespacing",
        # HTTP
        "allow-http",
        "http-address",
        "http-port",
        "reuse-http-port",
        "http-header-forward-pattern",
        "http-thread-count",
        "http-restricted-api",
        # GRPC
        "allow-grpc",
        "grpc-address",
        "grpc-port",
        "reuse-grpc-port",
        "grpc-header-forward-pattern",
        "grpc-infer-allocation-pool-size",
        "grpc-use-ssl",
        "grpc-use-ssl-mutual",
        "grpc-server-cert",
        "grpc-server-key",
        "grpc-root-cert",
        "grpc-infer-response-compression-level",
        "grpc-keepalive-time",
        "grpc-keepalive-timeout",
        "grpc-keepalive-permit-without-calls",
        "grpc-http2-max-pings-without-data",
        "grpc-http2-min-recv-ping-interval-without-data",
        "grpc-http2-max-ping-strikes",
        "grpc-max-connection-age",
        "grpc-max-connection-age-grace",
        "grpc-restricted-protocol",
        # Sagemaker
        "allow-sagemaker",
        "sagemaker-port",
        "sagemaker-safe-port-range",
        "sagemaker-thread-count",
        # Vertex
        "allow-vertex-ai",
        "vertex-ai-port",
        "vertex-ai-thread-count",
        "vertex-ai-default-model",
        # Metrics
        "allow-metrics",
        "allow-gpu-metrics",
        "allow-cpu-metrics",
        "metrics-address",
        "metrics-port",
        "metrics-interval-ms",
        "metrics-config",
        # Tracing
        "trace-config",
        # Backend
        "backend-directory",
        "backend-config",
        # Repository Agent
        "repoagent-directory",
        # Response Cache
        "cache-config",
        "cache-directory",
        # Rate Limiter
        "rate-limit",
        "rate-limit-resource",
        # Memory/Device Management
        "pinned-memory-pool-byte-size",
        "cuda-memory-pool-byte-size",
        "cuda-virtual-address-size",
        "min-supported-compute-capability",
        "buffer-management-thread-count",
        "host-policy",
        "model-load-gpu-limit",
        # DEPRECATED
        "strict-model-config",
        "response-cache-byte-size",
        "trace-file",
        "trace-level",
        "trace-rate",
        "trace-count",
        "trace-log-frequency",
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

        snake_cased_keys = [key.replace("-", "_") for key in cls.server_arg_keys]
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
                self[key.strip().replace("_", "-")] = params[key]

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

        return " ".join(
            [f"--{key}={val}" for key, val in self._server_args.items() if val]
        )

    def to_args_list(self):
        """
        Utility function to convert a cli string into a list of arguments while
        taking into account "smart" delimiters.  Notice in the example below
        that only the first equals sign is used as split delimiter.

        Returns
        -------
        list
            the list of arguments consisting of all set arguments to
            the tritonserver.

            Example:
            input cli_string: "--model-control-mode=explicit
                --backend-config=tensorflow,version=2"

            output: ['--model-control-mode', 'explicit',
                '--backend-config', 'tensorflow,version=2']
        """
        args_list = []
        args = self.to_cli_string().split()
        for arg in args:
            args_list += arg.split("=", 1)
        return args_list

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

        return self._server_args[key.strip().replace("_", "-")]

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

        kebab_cased_key = key.strip().replace("_", "-")
        if kebab_cased_key in self._server_args:
            self._server_args[kebab_cased_key] = value
        else:
            raise TritonModelAnalyzerException(
                f"The argument '{key}' to the Triton Inference "
                "Server is not supported by the model analyzer."
            )
