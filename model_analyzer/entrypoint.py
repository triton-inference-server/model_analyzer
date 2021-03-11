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

import sys
import os
import shutil
from prometheus_client.parser import text_string_to_metric_families
import requests
import numba.cuda
import signal
import logging
from urllib.parse import urlparse

from .cli.cli import CLI

from .analyzer import Analyzer
from .model_analyzer_exceptions import TritonModelAnalyzerException
from .triton.server.server_factory import TritonServerFactory
from .triton.server.server_config import TritonServerConfig
from .triton.client.client_factory import TritonClientFactory
from .output.file_writer import FileWriter
from .device.gpu_device_factory import GPUDeviceFactory
from .config.input.config import AnalyzerConfig

logger = logging.getLogger(__name__)
MAX_NUMBER_OF_INTERRUPTS = 3


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
        logger.info('Using remote Triton Server...')
        server = TritonServerFactory.create_server_local(path=None,
                                                         config=triton_config)
        logger.warn(
            'GPU memory metrics reported in the remote mode are not'
            ' accuracte. Model Analyzer uses Triton explicit model control to'
            ' load/unload models. Some frameworks do not release the GPU'
            ' memory even when the memory is not being used. Consider'
            ' using the "local" or "docker" mode if you want to accurately'
            ' monitor the GPU memory usage for different models.')
        logger.warn(
            'Config sweep parameters are ignored in the "remote" mode because'
            ' Model Analyzer does not have access to the model repository of'
            ' the remote Triton Server.')
    elif config.triton_launch_mode == 'local':
        triton_config = TritonServerConfig()
        triton_config.update_config(config.triton_server_flags)
        triton_config['model-repository'] = config.output_model_repository_path
        triton_config['http-port'] = config.triton_http_endpoint.split(':')[-1]
        triton_config['grpc-port'] = config.triton_grpc_endpoint.split(':')[-1]
        triton_config['metrics-port'] = urlparse(
            config.triton_metrics_url).port
        triton_config['model-control-mode'] = 'explicit'
        logger.info('Starting a local Triton Server...')
        server = TritonServerFactory.create_server_local(
            path=config.triton_server_path, config=triton_config)
    elif config.triton_launch_mode == 'docker':
        triton_config = TritonServerConfig()
        triton_config.update_config(config.triton_server_flags)
        triton_config['model-repository'] = os.path.abspath(
            config.output_model_repository_path)
        triton_config['http-port'] = config.triton_http_endpoint.split(':')[-1]
        triton_config['grpc-port'] = config.triton_grpc_endpoint.split(':')[-1]
        triton_config['metrics-port'] = urlparse(
            config.triton_metrics_url).port
        triton_config['model-control-mode'] = 'explicit'
        logger.info('Starting a Triton Server using docker...')
        server = TritonServerFactory.create_server_docker(
            image='nvcr.io/nvidia/tritonserver:' + config.triton_version,
            config=triton_config,
            gpus=get_analyzer_gpus(config))
    else:
        raise TritonModelAnalyzerException(
            f"Unrecognized triton-launch-mode : {config.triton_launch_mode}")

    return server


def get_analyzer_gpus(config):
    """
    Creates a list of GPU UUIDs corresponding to the GPUs visible to
    model_analyzer.

    Parameters
    ----------
    config : namespace
        The arguments passed into the CLI
    """

    model_analyzer_gpus = []
    if len(config.gpus) == 1 and config.gpus[0] == 'all':
        devices = numba.cuda.list_devices()
        for device in devices:
            gpu_device = GPUDeviceFactory.create_device_by_cuda_index(
                device.id)
            model_analyzer_gpus.append(
                str(gpu_device.device_uuid(), encoding='ascii'))
    else:
        devices = config.gpus
        for device in devices:
            gpu_device = GPUDeviceFactory.create_device_by_uuid(device)
            model_analyzer_gpus.append(
                str(gpu_device.device_uuid(), encoding='ascii'))

    return model_analyzer_gpus


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


def check_triton_and_model_analyzer_gpus(config):
    """
    Check whether Triton Server and Model Analyzer are using the same GPUs

    Parameters
    ----------
    config : namespace
        The arguments passed into the CLI

    Raises
    ------
    TritonModelAnalyzerException
        If they are using different GPUs this exception will be raised.
    """

    model_analyzer_gpus = get_analyzer_gpus(config)
    triton_gpus = get_triton_metrics_gpus(config)
    if set(model_analyzer_gpus) != set(triton_gpus):
        raise TritonModelAnalyzerException(
            "'Triton Server is not using the same GPUs as Model Analyzer: '"
            f"Model Analyzer GPUs {model_analyzer_gpus}, Triton GPUs {triton_gpus}"
        )


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


def write_server_logs(config, server):
    """
    Checks if server logs have been
    requested, and writes them
    to the specified file

    Parameters
    ----------
    config : namespace
        The arguments passed into the CLI
    server : TritonServer
        The triton server instance whose logs
        we may want to write out.
    """

    if config.triton_output_path:
        server_log_writer = FileWriter(filename=config.triton_output_path)
        server_log_writer.write(server.logs())


def interrupt_handler(signal, frame):
    """
    A signal handler to properly
    shutdown the model analyzer on
    interrupt
    """

    global exiting
    exiting += 1
    logging.info(
        f'Received SIGINT. Exiting ({exiting}/{MAX_NUMBER_OF_INTERRUPTS})...')

    if exiting == MAX_NUMBER_OF_INTERRUPTS:
        sys.exit(1)
    return


def main():
    """
    Main entrypoint of model_analyzer
    """

    global exiting

    # Number of Times User Requested Exit
    exiting = 0

    metric_tags = [
        "perf_throughput", "perf_latency", "gpu_used_memory",
        "gpu_free_memory", "gpu_utilization", "cpu_used_ram",
        "cpu_available_ram"
    ]

    signal.signal(signal.SIGINT, interrupt_handler)

    try:
        config = AnalyzerConfig()
        cli = CLI(config)
        cli.parse()
    except TritonModelAnalyzerException as e:
        logging.error(f'Model Analyzer encountered an error: {e}')
        sys.exit(1)

    logging.info(
        f'Triton Model Analyzer started: config={config.get_all_config()}')

    server = None
    # Generate the output model repository path folder.
    output_model_repo_path = config.output_model_repository_path
    try:
        os.mkdir(output_model_repo_path)
    except OSError:
        if not config.override_output_model_repository:
            raise TritonModelAnalyzerException(
                f'Path "{output_model_repo_path}" already exists. '
                'Please set or modify "--output-model-repository-path" flag or remove this directory.'
                ' You can also allow overriding of the output directory using'
                ' the "--override-output-model-repository" flag.')
        else:
            shutil.rmtree(output_model_repo_path)
            logger.warn(
                f'Overriding the output model repo path "{output_model_repo_path}"...'
            )
            os.mkdir(output_model_repo_path)
    try:
        client, server = get_triton_handles(config)

        # Only check for exit after the events that take a long time.
        if exiting:
            return

        analyzer = Analyzer(config, client, metric_tags, server)

        # Check TritonServer GPUs
        server.start()
        client.wait_for_server_ready(config.max_retries)
        check_triton_and_model_analyzer_gpus(config)
        server.stop()

        if exiting:
            return

        logging.info('Starting perf_analyzer...')
        analyzer.run()
        analyzer.write_and_export_results()
    finally:
        if server is not None:
            server.stop()
            write_server_logs(config, server)


if __name__ == '__main__':
    main()
