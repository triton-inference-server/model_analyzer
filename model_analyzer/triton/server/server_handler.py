# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.server.server_factory import TritonServerFactory

from model_analyzer.config.input.config_utils import binary_path_validator

from model_analyzer.constants import CONFIG_PARSER_FAILURE, LOGGER_NAME

from urllib.parse import urlparse

import os
import logging

logger = logging.getLogger(LOGGER_NAME)


class TritonServerHandler():
    """
    Static methods used to create and return Triton Servers
    """

    def __init__(self):
        NotImplemented

    @staticmethod
    def get_server_handle(config, gpus, use_model_repository=False):
        """
        Creates and returns a TritonServer
        with specified arguments

        Parameters
        ----------
        config : namespace
            Arguments parsed from the CLI
        gpus : list of str
            Available, supported, visible requested GPU UUIDs
        use_model_repository : bool
            Optional flag to use the model directory instead
            of the output model directory
        Returns
        -------
        TritonServer
            Handle to the Triton Server
        """

        if config.triton_launch_mode == 'remote':
            server = TritonServerHandler._get_remote_server_handle(config)
        elif config.triton_launch_mode == 'local':
            server = TritonServerHandler._get_local_server_handle(
                config, gpus, use_model_repository)
        elif config.triton_launch_mode == 'docker':
            server = TritonServerHandler._get_docker_server_handle(
                config, gpus, use_model_repository)
        elif config.triton_launch_mode == 'c_api':
            server = TritonServerHandle._get_c_api_server_handle(
                config, use_model_repository)
        else:
            raise TritonModelAnalyzerException(
                f"Unrecognized triton-launch-mode : {config.triton_launch_mode}"
            )

        return server

    @staticmethod
    def _get_remote_server_handle(config):
        triton_config = TritonServerConfig()
        triton_config.update_config(config.triton_server_flags)
        triton_config['model-repository'] = 'remote-model-repository'
        logger.info('Using remote Triton Server')
        server = TritonServerFactory.create_server_local(path=None,
                                                         config=triton_config,
                                                         gpus=[],
                                                         log_path="")
        logger.warning(
            'GPU memory metrics reported in the remote mode are not'
            ' accurate. Model Analyzer uses Triton explicit model control to'
            ' load/unload models. Some frameworks do not release the GPU'
            ' memory even when the memory is not being used. Consider'
            ' using the "local" or "docker" mode if you want to accurately'
            ' monitor the GPU memory usage for different models.')
        logger.warning(
            'Config sweep parameters are ignored in the "remote" mode because'
            ' Model Analyzer does not have access to the model repository of'
            ' the remote Triton Server.')

        return server

    @staticmethod
    def _get_local_server_handle(config, gpus, use_model_repository):
        TritonServerHandler._validate_triton_server_path(config)

        triton_config = TritonServerConfig()
        triton_config.update_config(config.triton_server_flags)

        triton_config[
            'model-repository'] = config.model_repository if use_model_repository \
                                                            else config.output_model_repository_path

        triton_config['http-port'] = config.triton_http_endpoint.split(':')[-1]
        triton_config['grpc-port'] = config.triton_grpc_endpoint.split(':')[-1]
        triton_config['metrics-port'] = urlparse(config.triton_metrics_url).port
        triton_config['model-control-mode'] = 'explicit'
        if config.use_local_gpu_monitor:
            triton_config['metrics-interval-ms'] = int(
                config.monitoring_interval * 1e3)
        logger.info('Starting a local Triton Server')
        server = TritonServerFactory.create_server_local(
            path=config.triton_server_path,
            config=triton_config,
            gpus=gpus,
            log_path=config.triton_output_path)

        return server

    @staticmethod
    def _get_docker_server_handle(config, gpus, use_model_repository):
        triton_config = TritonServerConfig()
        triton_config.update_config(config.triton_server_flags)

        if (use_model_repository):
            triton_config['model-repository'] = os.path.abspath(
                config.model_repository)
        else:
            triton_config['model-repository'] = os.path.abspath(
                config.output_model_repository_path)

        triton_config['http-port'] = config.triton_http_endpoint.split(':')[-1]
        triton_config['grpc-port'] = config.triton_grpc_endpoint.split(':')[-1]
        triton_config['metrics-port'] = urlparse(config.triton_metrics_url).port
        triton_config['model-control-mode'] = 'explicit'
        if config.use_local_gpu_monitor:
            triton_config['metrics-interval-ms'] = int(
                config.monitoring_interval * 1e3)
        logger.info('Starting a Triton Server using docker')
        server = TritonServerFactory.create_server_docker(
            image=config.triton_docker_image,
            config=triton_config,
            gpus=gpus,
            log_path=config.triton_output_path,
            mounts=config.triton_docker_mounts,
            labels=config.triton_docker_labels,
            shm_size=config.triton_docker_shm_size)

        return server

    @staticmethod
    def _get_c_api_server_handle(config, use_model_repository):
        TritonServerHandler._validate_triton_install_path(config)

        triton_config = TritonServerConfig()

        if (use_model_repository):
            triton_config['model-repository'] = os.path.abspath(
                config.model_repository)
        else:
            triton_config['model-repository'] = os.path.abspath(
                config.output_model_repository_path)

        logger.info("Starting a Triton Server using perf_analyzer's C_API")
        server = TritonServerFactory.create_server_local(path=None,
                                                         config=triton_config,
                                                         gpus=[],
                                                         log_path="")
        logger.warning(
            "When profiling with perf_analyzer's C_API, some metrics may be "
            "affected. Triton is not launched with explicit model control "
            "mode, and as a result, loads all model config variants as they "
            "are created in the output_model_repository.")

        return server

    @staticmethod
    def _validate_triton_server_path(config):
        """
        Raises an execption if 'triton_server_path' doesn't exist

        Parameters
        ----------
        config : namespace
            Arguments parsed from the CLI
        """
        path = config.get_config()['triton_server_path'].value()
        config_status = binary_path_validator(path)
        if config_status.status() == CONFIG_PARSER_FAILURE:
            raise TritonModelAnalyzerException(config_status.message())

    @staticmethod
    def _validate_triton_install_path(config):
        """
        Raises an exception in the following cases: 
          - 'triton_install_path' doesn't exist
          - 'trtion_install_path' exists, but contains no files
        
        Parameters
        ----------
        config : namespace
            Arguments parsed from the CLI
        """
        path = config.get_config()['triton_install_path'].value()

        # Check the file system
        if not path or not os.path.exists(path) or not os.path.isdir(path):
            raise TritonModelAnalyzerException(
                f"triton_install_path {path} is not specified, does not exist, " \
                "or is not a directory."
            )

        # Make sure that files exist in the install directory
        if len(os.listdir(path)) == 0:
            raise TritonModelAnalyzerException(
                f"triton_install_path {path} should not be empty.")
