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
        'log_verbose',
        'log_info',
        'log_warning',
        'log_error',
        'id',
        # Model Repository
        'model_store',
        'model_repository',
        # Exit
        'exit_timeout_secs',
        'exit_on_error',
        # Strictness
        'strict_model_config',
        'strict_readiness',
        # API Servers
        'allow_http',
        'http_port',
        'http_thread_count',
        'allow_grpc',
        'grpc_port',
        'grpc_infer_allocation_pool_size',
        'grpc_use_ssl',
        'grpc_server_cert',
        'grpc_server_key',
        'grpc_root_cert',
        'allow_metrics',
        'allow_gpu_metrics',
        'metrics_port',
        # Tracing
        'trace_file',
        'trace_level',
        'trace_rate',
        # Model control
        'model_control_mode',
        'repository_poll_secs',
        'load_model',
        # Memory and GPU
        'pinned_memory_pool_byte_size',
        'cuda_memory_pool_byte_size',
        'min_supported_compute_capability',
        # Backend config
        'backend_directory',
        'backend_config',
        'allow_soft_placement',
        'gpu_memory_fraction',
        'tensorflow_version'
    ]

    def __init__(self):
        """
        Construct TritonServerConfig
        """

        self._server_args = {k: None for k in self.server_arg_keys}

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
                self[key.strip().replace('-', '_')] = params[key]

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
            f'--{key.strip().replace("_", "-")}={val}'
            for key, val in self._server_args.items() if val
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

        return self._server_args[key.strip().replace('-', '_')]

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

        snake_case_key = key.strip().replace('-', '_')
        if snake_case_key in self._server_args:
            self._server_args[snake_case_key] = value
        else:
            raise TritonModelAnalyzerException(
                f"The argument '{key}' to the Triton Inference "
                "Server is not supported by the model analyzer.")
