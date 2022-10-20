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

from model_analyzer.device.gpu_device_factory import GPUDeviceFactory
from .analyzer import Analyzer
from .cli.cli import CLI
from .model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.constants import LOGGER_NAME
from .triton.client.client_factory import TritonClientFactory
from .triton.server.server_factory import TritonServerFactory
from .state.analyzer_state_manager import AnalyzerStateManager
from .config.input.config_command_profile import ConfigCommandProfile
from .config.input.config_command_report import ConfigCommandReport
from .log_formatter import setup_logging
import sys
import os
import logging
import shutil
from pprint import pformat

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
        http_ssl_options = get_http_ssl_options(config)
        client = TritonClientFactory.create_http_client(
            server_url=config.triton_http_endpoint,
            ssl_options=http_ssl_options)
    elif config.client_protocol == 'grpc':
        grpc_ssl_options = get_grpc_ssl_options(config)
        client = TritonClientFactory.create_grpc_client(
            server_url=config.triton_grpc_endpoint,
            ssl_options=grpc_ssl_options)
    else:
        raise TritonModelAnalyzerException(
            f"Unrecognized client-protocol : {config.client_protocol}")

    return client


def get_http_ssl_options(config):
    """
    Returns HTTP SSL options dictionary

    Parameters
    ----------
    config : namespace
        Arguments parsed from the CLI
    """

    ssl_option_keys = [
        'ssl-https-verify-peer', 'ssl-https-verify-host',
        'ssl-https-ca-certificates-file', 'ssl-https-client-certificate-file',
        'ssl-https-client-certificate-type', 'ssl-https-private-key-file',
        'ssl-https-private-key-type'
    ]

    return {
        key: config.perf_analyzer_flags[key]
        for key in ssl_option_keys
        if key in config.perf_analyzer_flags
    }


def get_grpc_ssl_options(config):
    """
    Returns gRPC SSL options dictionary

    Parameters
    ----------
    config : namespace
        Arguments parsed from the CLI
    """

    ssl_option_keys = [
        'ssl-grpc-use-ssl',
        'ssl-grpc-root-certifications-file',
        'ssl-grpc-private-key-file',
        'ssl-grpc-certificate-chain-file',
    ]

    return {
        key: config.perf_analyzer_flags[key]
        for key in ssl_option_keys
        if key in config.perf_analyzer_flags
    }


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
    fail_if_server_already_running(client, config)
    server = TritonServerFactory.get_server_handle(config, gpus)

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
            'DEPRECATED: Aliased to profile - please use profile subcommand.',
            config=config_profile)
        cli.add_subcommand(cmd='report',
                           help='Generate detailed reports for a single config',
                           config=config_report)
        return cli.parse()

    except TritonModelAnalyzerException as e:
        logger.error(f'Model Analyzer encountered an error: {e}')
        sys.exit(1)


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
                           f'"{config.output_model_repository_path}"')
            os.mkdir(config.output_model_repository_path)


def fail_if_server_already_running(client, config):
    """ 
    Checks if there is already a Triton server running
    If there is and the launch mode is not 'remote' or 'c_api', throw an exception
    Else, nothing will happen
    """
    if config.triton_launch_mode == 'remote' or config.triton_launch_mode == "c_api":
        return

    is_server_running = True
    try:
        client.is_server_ready()
    except:
        is_server_running = False
    finally:
        if is_server_running:
            raise TritonModelAnalyzerException(
                f"Another application (likely a Triton Server) is already using the desired port. In '{config.triton_launch_mode}' mode, Model Analyzer will launch a Triton Server and requires that the HTTP/GRPC port is not occupied by another application. Please kill the other application or specify a different port."
            )


def main():
    """
    Main entrypoint of model_analyzer
    """

    # Need to create a basic logging format for logs we print
    # before we have enough information to configure the full logger
    logging.basicConfig(format="[Model Analyzer] %(message)s")

    args, config = get_cli_and_config_options()

    setup_logging(quiet=args.quiet, verbose=args.verbose)

    logger.debug("\n%s", pformat(config.get_all_config()))

    # Launch subcommand handlers
    server = None
    try:
        # Make calls to correct analyzer subcommand functions
        if args.subcommand == 'profile' or args.subcommand == 'analyze':

            if args.subcommand == 'profile' and not config.model_repository:
                raise TritonModelAnalyzerException(
                    "No model repository specified. Please specify it using the YAML config file or using the --model-repository flag in CLI."
                )

            # Set up devices
            gpus = GPUDeviceFactory().verify_requested_gpus(config.gpus)

            # Check/create output model repository
            create_output_model_repository(config)

            client, server = get_triton_handles(config, gpus)
            state_manager = AnalyzerStateManager(config=config, server=server)

            # Only check for exit after the events that take a long time.
            if state_manager.exiting():
                return

            analyzer = Analyzer(config,
                                server,
                                state_manager,
                                checkpoint_required=False)
            analyzer.profile(client=client,
                             gpus=gpus,
                             mode=args.mode,
                             verbose=args.verbose)
        elif args.subcommand == 'report':

            analyzer = Analyzer(config,
                                server,
                                AnalyzerStateManager(config=config,
                                                     server=server),
                                checkpoint_required=True)
            analyzer.report(mode=args.mode)
    finally:
        if server is not None:
            server.stop()


if __name__ == '__main__':
    main()
