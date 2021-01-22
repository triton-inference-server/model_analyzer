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

import os
import sys
from itertools import product
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
from .triton.model.model import Model
from .record.metrics_mapper import MetricsMapper
from .output.file_writer import FileWriter
from .device.gpu_device_factory import GPUDeviceFactory
from .config.config import AnalyzerConfig

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
        triton_config['model-repository'] = 'remote-model-repository'
        logger.info('Using remote Triton Server...')
        server = TritonServerFactory.create_server_local(path=None,
                                                         config=triton_config)
    elif config.triton_launch_mode == 'local':
        triton_config = TritonServerConfig()
        triton_config['model-repository'] = config.model_repository
        triton_config['http-port'] = config.triton_http_endpoint.split(':')[-1]
        triton_config['grpc-port'] = config.triton_grpc_endpoint.split(':')[-1]
        triton_config['metrics-port'] = urlparse(
            config.triton_metrics_url).port
        triton_config['model-control-mode'] = 'explicit'
        logger.info('Starting a local Triton Server...')
        server = TritonServerFactory.create_server_local(
            path=config.triton_server_path, config=triton_config)
        server.start()
    elif config.triton_launch_mode == 'docker':
        triton_config = TritonServerConfig()
        triton_config['model-repository'] = config.model_repository
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
        server.start()
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

    if len(config.gpus) == 1 and config.gpus[0] == 'all':
        devices = numba.cuda.list_devices()
    else:
        devices = config.gpus

    model_analyzer_gpus = []
    for device in devices:
        gpu_device = GPUDeviceFactory.create_device_by_cuda_index(device.id)
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
    client.wait_for_server_ready(num_retries=config.max_retries)
    logger.info('Triton Server is ready.')

    return client, server


def create_run_configs(config):
    """
    Parameters
    ----------
    config : namespace
        The arguments passed into the CLI

    Returns
    -------
    list of dicts
        keys are parameters to perf_analyzer
        values are individual combinations of argument values
    """

    sweep_params = {
        'model-name': config.model_names,
        'batch-size': config.batch_sizes,
        'concurrency-range': config.concurrency,
        'protocol': [config.client_protocol],
        'url': [
            config.triton_http_endpoint if config.client_protocol == 'http'
            else config.triton_grpc_endpoint
        ],
        'measurement-interval': [config.perf_measurement_window]
    }
    param_combinations = list(product(*tuple(sweep_params.values())))
    run_params = [
        dict(zip(sweep_params.keys(), vals)) for vals in param_combinations
    ]

    return run_params


def write_results(config, analyzer):
    """
    Makes calls to the analyzer to write results out to streams or files. If
    exporting results is requested, uses a FileWriter for specified output
    files.

    Parameters
    ----------
    config : namespace
        The arguments passed into the CLI
    analyzer : Analyzer
        The instance being used to profile
        server inferencing.
    """

    analyzer.write_results(writer=FileWriter(), column_separator=' ')
    if config.export:
        server_metrics_path = os.path.join(config.export_path,
                                           config.filename_server_only)
        analyzer.export_server_only_csv(
            writer=FileWriter(filename=server_metrics_path),
            column_separator=',')
        metrics_inference_path = os.path.join(config.export_path,
                                              config.filename_model_inference)
        metrics_gpu_path = os.path.join(config.export_path,
                                        config.filename_model_gpu)
        analyzer.export_model_csv(
            inference_writer=FileWriter(filename=metrics_inference_path),
            gpu_metrics_writer=FileWriter(filename=metrics_gpu_path),
            column_separator=',')


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


def run_analyzer(config, analyzer, client, run_configs):
    """
    Makes a single call to profile the server only Then for each run
    configurations, it profiles model inference.

    Parameters
    ----------
    config : namespace
        The arguments passed into the CLI
    analyzer : Analyzer
        The instance being used to profile server inferencing.
    client : TritonClient
        Instance used to load/unload models
    run_configs : list of dicts
        Output of create_run_configs

    Raises
    ------
    TritonModelAnalyzerException
    """

    analyzer.profile_server_only()
    for run_config in run_configs:
        model = Model(name=run_config['model-name'])
        client.load_model(model=model)
        client.wait_for_model_ready(model=model,
                                    num_retries=config.max_retries)
        try:
            perf_output_writer = None if config.no_perf_output else FileWriter(
            )
            analyzer.profile_model(run_config=run_config,
                                   perf_output_writer=perf_output_writer)
        finally:
            client.unload_model(model=model)


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

    monitoring_metrics = MetricsMapper.get_monitoring_metrics(metric_tags)

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
    try:
        client, server = get_triton_handles(config)

        # Only check for exit after the events that take a long time.
        if exiting:
            return

        analyzer = Analyzer(config, monitoring_metrics, server)
        check_triton_and_model_analyzer_gpus(config)
        run_configs = create_run_configs(config)
        if exiting:
            return

        logging.info('Starting perf_analyzer...')
        run_analyzer(config, analyzer, client, run_configs)
        write_results(config, analyzer)
    except TritonModelAnalyzerException as e:
        logging.exception(f'Model Analyzer encountered an error: {e}')
    finally:
        if server is not None:
            server.stop()
            write_server_logs(config, server)


if __name__ == '__main__':
    main()
