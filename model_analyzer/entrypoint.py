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

from model_analyzer.device.gpu_device_factory import GPUDeviceFactory
from .analyzer import Analyzer
from .cli.cli import CLI
from .model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.constants import LOGGER_NAME
from .triton.server.server_factory import TritonServerFactory
from .triton.server.server_config import TritonServerConfig
from .triton.client.client_factory import TritonClientFactory
from .state.analyzer_state_manager import AnalyzerStateManager
from .config.input.config_command_profile import ConfigCommandProfile
from .config.input.config_command_analyze import ConfigCommandAnalyze
from .config.input.config_command_report import ConfigCommandReport

import sys
import os
import logging
import shutil
from urllib.parse import urlparse

logger = logging.getLogger(LOGGER_NAME)


def get_client_handle(config):
    """
    Creates and returns a TritonClient
    with specified arguments

    Parameters
    ----------
    config : namespace
        Arguments parsed from the CLI
    """

    if config.client_protocol == 'http':
        client = TritonClientFactory.create_http_client(
            server_url=config.triton_http_endpoint)
    elif config.client_protocol == 'grpc':
        client = TritonClientFactory.create_grpc_client(
            server_url=config.triton_grpc_endpoint)
    else:
        raise TritonModelAnalyzerException(
            f"Unrecognized client-protocol : {config.client_protocol}")

    return client


def get_server_handle(config, gpus):
    """
    Creates and returns a TritonServer
    with specified arguments

    Parameters
    ----------
    config : namespace
        Arguments parsed from the CLI
    gpus : list of str
        Available, supported, visible requested GPU UUIDs
    """

    if config.triton_launch_mode == 'remote':
        triton_config = TritonServerConfig()
        triton_config.update_config(config.triton_server_flags)
        triton_config['model-repository'] = 'remote-model-repository'
        logger.info('Using remote Triton Server...')
        server = TritonServerFactory.create_server_local(path=None,
                                                         config=triton_config,
                                                         gpus=[],
                                                         log_path="")
        logger.warning(
            'GPU memory metrics reported in the remote mode are not'
            ' accuracte. Model Analyzer uses Triton explicit model control to'
            ' load/unload models. Some frameworks do not release the GPU'
            ' memory even when the memory is not being used. Consider'
            ' using the "local" or "docker" mode if you want to accurately'
            ' monitor the GPU memory usage for different models.')
        logger.warning(
            'Config sweep parameters are ignored in the "remote" mode because'
            ' Model Analyzer does not have access to the model repository of'
            ' the remote Triton Server.')
    elif config.triton_launch_mode == 'local':
        triton_config = TritonServerConfig()
        triton_config.update_config(config.triton_server_flags)
        triton_config['model-repository'] = config.output_model_repository_path
        triton_config['http-port'] = config.triton_http_endpoint.split(':')[-1]
        triton_config['grpc-port'] = config.triton_grpc_endpoint.split(':')[-1]
        triton_config['metrics-port'] = urlparse(config.triton_metrics_url).port
        triton_config['model-control-mode'] = 'explicit'
        if config.use_local_gpu_monitor:
            triton_config['metrics-interval-ms'] = int(
                config.monitoring_interval * 1e3)
        logger.info('Starting a local Triton Server...')
        server = TritonServerFactory.create_server_local(
            path=config.triton_server_path,
            config=triton_config,
            gpus=gpus,
            log_path=config.triton_output_path)
    elif config.triton_launch_mode == 'docker':
        triton_config = TritonServerConfig()
        triton_config.update_config(config.triton_server_flags)
        triton_config['model-repository'] = os.path.abspath(
            config.output_model_repository_path)
        triton_config['http-port'] = config.triton_http_endpoint.split(':')[-1]
        triton_config['grpc-port'] = config.triton_grpc_endpoint.split(':')[-1]
        triton_config['metrics-port'] = urlparse(config.triton_metrics_url).port
        triton_config['model-control-mode'] = 'explicit'
        if config.use_local_gpu_monitor:
            triton_config['metrics-interval-ms'] = int(
                config.monitoring_interval * 1e3)
        logger.info('Starting a Triton Server using docker...')
        server = TritonServerFactory.create_server_docker(
            image=config.triton_docker_image,
            config=triton_config,
            gpus=gpus,
            log_path=config.triton_output_path,
            mounts=config.triton_docker_mounts,
            labels=config.triton_docker_labels)
    elif config.triton_launch_mode == 'c_api':
        triton_config = TritonServerConfig()
        triton_config['model-repository'] = os.path.abspath(
            config.output_model_repository_path)
        logger.info("Starting a Triton Server using perf_analyzer's C_API...")
        server = TritonServerFactory.create_server_local(path=None,
                                                         config=triton_config,
                                                         gpus=[],
                                                         log_path="")
        logger.warning(
            "When profiling with perf_analyzer's C_API, some metrics may be "
            "affected. Triton is not launched with explicit model control "
            "mode, and as a result, loads all model config variants as they "
            "are created in the output_model_repository.")
    else:
        raise TritonModelAnalyzerException(
            f"Unrecognized triton-launch-mode : {config.triton_launch_mode}")

    return server


def get_triton_handles(config, gpus):
    """
    Creates a TritonServer and starts it. Creates a TritonClient

    Parameters
    ----------
    config : namespace
        The arguments passed into the CLI
    gpus : list of str
        Available, supported, visible requested GPU UUIDs

    Returns
    -------
    TritonClient, TritonServer
        Handles for triton client/server pair.
    """

    client = get_client_handle(config)
    server = get_server_handle(config, gpus)

    return client, server


def get_cli_and_config_options():
    """
    Parses CLI/Yaml Config file options
    into Namespace and Config objects for
    the correct subcommand

    Returns
    -------
    args : Namespace
        Object that contains the parse CLI commands
        Used for the global options
    config: CommandConfig
        The config corresponding to the command being run,
        already filled in with values from CLI or YAML.
    """

    # Parse CLI options
    try:
        config_profile = ConfigCommandProfile()
        config_analyze = ConfigCommandAnalyze()
        config_report = ConfigCommandReport()

        cli = CLI()
        cli.add_subcommand(
            cmd='profile',
            help=
            'Run model inference profiling based on specified CLI or config options.',
            config=config_profile)
        cli.add_subcommand(
            cmd='analyze',
            help=
            'Collect and sort profiling results and generate data and summaries.',
            config=config_analyze)
        cli.add_subcommand(cmd='report',
                           help='Generate detailed reports for a single config',
                           config=config_report)
        return cli.parse()

    except TritonModelAnalyzerException as e:
        logger.error(f'Model Analyzer encountered an error: {e}')
        sys.exit(1)


def setup_logging(args):
    """
    Setup logger format

    Parameters
    ----------
    args : Namespace
        Contains arguments for verbosity of Model Analyzer
    """

    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level=log_level)


def create_output_model_repository(config):
    """
    Creates output model repository

    Parameters
    ----------
    ConfigCommandProfile
        The config containing the output_model_repository_path
    """

    try:
        os.mkdir(config.output_model_repository_path)
    except FileExistsError:
        if not config.override_output_model_repository:
            raise TritonModelAnalyzerException(
                f'Path "{config.output_model_repository_path}" already exists. '
                'Please set or modify "--output-model-repository-path" flag or remove this directory.'
                ' You can also allow overriding of the output directory using'
                ' the "--override-output-model-repository" flag.')
        else:
            shutil.rmtree(config.output_model_repository_path)
            logger.warning('Overriding the output model repo path '
                           f'"{config.output_model_repository_path}"...')
            os.mkdir(config.output_model_repository_path)


def main():
    """
    Main entrypoint of model_analyzer
    """

    # Configs and logging
    logging.basicConfig(format="%(asctime)s.%(msecs)d %(levelname)-4s"
                        "[%(filename)s:%(lineno)d] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    args, config = get_cli_and_config_options()
    setup_logging(args)

    logger.debug(config.get_all_config())

    # Launch subcommand handlers
    server = None
    try:
        # Make calls to correct analyzer subcommand functions
        if args.subcommand == 'profile':

            # Set up devices
            gpus = GPUDeviceFactory().verify_requested_gpus(config.gpus)

            # Check/create output model repository
            create_output_model_repository(config)

            client, server = get_triton_handles(config, gpus)
            state_manager = AnalyzerStateManager(config=config, server=server)

            # Only check for exit after the events that take a long time.
            if state_manager.exiting():
                return

            analyzer = Analyzer(config, server, state_manager)
            analyzer.profile(client=client, gpus=gpus)

        elif args.subcommand == 'analyze':

            analyzer = Analyzer(
                config, server,
                AnalyzerStateManager(config=config, server=server))
            analyzer.analyze(mode=args.mode, quiet=bool(args.quiet))
        elif args.subcommand == 'report':

            analyzer = Analyzer(
                config, server,
                AnalyzerStateManager(config=config, server=server))
            analyzer.report(mode=args.mode)
    finally:
        if server is not None:
            server.stop()


if __name__ == '__main__':
    main()
