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

from .analyzer import Analyzer
from .cli.cli import CLI
from .model_analyzer_exceptions import TritonModelAnalyzerException
from .triton.server.server_factory import TritonServerFactory
from .triton.server.server_config import TritonServerConfig
from .triton.client.client_factory import TritonClientFactory
from .device.gpu_device_factory import GPUDeviceFactory
from .state.analyzer_state_manager import AnalyzerStateManager
from .config.input.config_command_profile import ConfigCommandProfile
from .config.input.config_command_analyze import ConfigCommandAnalyze
from .config.input.config_command_report import ConfigCommandReport

import sys
import os
from prometheus_client.parser import text_string_to_metric_families
import requests
import logging
import shutil
from urllib.parse import urlparse


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


def get_server_handle(config):
    """
    Creates and returns a TritonServer
    with specified arguments

    Parameters
    ----------
    config : namespace
        Arguments parsed from the CLI
    """

    if config.triton_launch_mode == 'remote':
        triton_config = TritonServerConfig()
        triton_config.update_config(config.triton_server_flags)
        triton_config['model-repository'] = 'remote-model-repository'
        logging.info('Using remote Triton Server...')
        server = TritonServerFactory.create_server_local(path=None,
                                                         config=triton_config)
        logging.warn(
            'GPU memory metrics reported in the remote mode are not'
            ' accuracte. Model Analyzer uses Triton explicit model control to'
            ' load/unload models. Some frameworks do not release the GPU'
            ' memory even when the memory is not being used. Consider'
            ' using the "local" or "docker" mode if you want to accurately'
            ' monitor the GPU memory usage for different models.')
        logging.warn(
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
        logging.info('Starting a local Triton Server...')
        server = TritonServerFactory.create_server_local(
            path=config.triton_server_path, config=triton_config)
    elif config.triton_launch_mode == 'docker':
        triton_config = TritonServerConfig()
        triton_config.update_config(config.triton_server_flags)
        triton_config['model-repository'] = os.path.abspath(
            config.output_model_repository_path)
        triton_config['http-port'] = config.triton_http_endpoint.split(':')[-1]
        triton_config['grpc-port'] = config.triton_grpc_endpoint.split(':')[-1]
        triton_config['metrics-port'] = urlparse(config.triton_metrics_url).port
        triton_config['model-control-mode'] = 'explicit'
        logging.info('Starting a Triton Server using docker...')
        server = TritonServerFactory.create_server_docker(
            image=config.triton_docker_image,
            config=triton_config,
            gpus=GPUDeviceFactory.get_analyzer_gpus(config.gpus))
    else:
        raise TritonModelAnalyzerException(
            f"Unrecognized triton-launch-mode : {config.triton_launch_mode}")

    return server


def get_triton_metrics_gpus(config):
    """
    Uses prometheus to request a list of GPU UUIDs corresponding to the GPUs
    visible to Triton Inference Server

    Parameters
    ----------
    config : namespace
        The arguments passed into the CLI
    """

    triton_prom_str = str(requests.get(config.triton_metrics_url).content,
                          encoding='ascii')
    metrics = text_string_to_metric_families(triton_prom_str)

    triton_gpus = []
    for metric in metrics:
        if metric.name == 'nv_gpu_utilization':
            for sample in metric.samples:
                triton_gpus.append(sample.labels['gpu_uuid'])

    return triton_gpus


def check_triton_and_model_analyzer_gpus(client, server, config):
    """
    Check whether Triton Server and Model Analyzer are using the same GPUs

    Parameters
    ----------
    client: TritonClient
        Handle for client
    server: TritonServer
        Handle for server
    config : namespace
        The arguments passed into the CLI

    Raises
    ------
    TritonModelAnalyzerException
        If they are using different GPUs this exception will be raised.
    """

    server.start()
    client.wait_for_server_ready(config.max_retries)

    model_analyzer_gpus = GPUDeviceFactory.get_analyzer_gpus(config.gpus)
    triton_gpus = get_triton_metrics_gpus(config)
    if set(model_analyzer_gpus) != set(triton_gpus):
        raise TritonModelAnalyzerException(
            "'Triton Server is not using the same GPUs as Model Analyzer: '"
            f"Model Analyzer GPUs {model_analyzer_gpus}, Triton GPUs {triton_gpus}"
        )

    server.stop()


def get_triton_handles(config):
    """
    Creates a TritonServer and starts it. Creates a TritonClient

    Parameters
    ----------
    config : namespace
        The arguments passed into the CLI

    Returns
    -------
    TritonClient, TritonServer
        Handles for triton client/server pair.
    """

    client = get_client_handle(config)
    server = get_server_handle(config)

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
        logging.error(f'Model Analyzer encountered an error: {e}')
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
    logging.basicConfig(level=log_level,
                        format="%(asctime)s.%(msecs)d %(levelname)-4s"
                        "[%(filename)s:%(lineno)d] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")


def create_output_model_repository(config):
    """
    Creates 

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
            logging.warn('Overriding the output model repo path '
                         f'"{config.output_model_repository_path}"...')
            os.mkdir(config.output_model_repository_path)


def main():
    """
    Main entrypoint of model_analyzer
    """

    args, config = get_cli_and_config_options()
    setup_logging(args)
    state_manager = AnalyzerStateManager(config=config)

    server = None
    try:
        # Make calls to correct analyzer subcommand functions
        if args.subcommand == 'profile':
            # Check/create output model repository
            create_output_model_repository(config)

            client, server = get_triton_handles(config)
            # Only check for exit after the events that take a long time.
            if state_manager.exiting():
                return
            check_triton_and_model_analyzer_gpus(client, server, config)
            if state_manager.exiting():
                return

            analyzer = Analyzer(config, server, state_manager)
            analyzer.profile(client=client)

        elif args.subcommand == 'analyze':

            analyzer = Analyzer(config, server, state_manager)
            analyzer.analyze()
        elif args.subcommand == 'report':

            analyzer = Analyzer(config, server, state_manager)
            analyzer.report()
    finally:
        if server is not None:
            server.stop()


if __name__ == '__main__':
    main()
