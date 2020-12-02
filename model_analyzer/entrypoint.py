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
from .record.gpu_free_memory import GPUFreeMemory
from .record.gpu_used_memory import GPUUsedMemory
from .record.gpu_utilization import GPUUtilization
from .record.perf_throughput import PerfThroughput
from .record.perf_latency import PerfLatency
from .output.file_writer import FileWriter
from .device.gpu_device_factory import GPUDeviceFactory

logger = logging.getLogger(__name__)
MAX_NUMBER_OF_INTERRUPTS = 3


def get_client_handle(args):
    """
    Creates and returns a TritonClient
    with specified arguments

    Parameters
    ----------
    args : namespace
        Arguments parsed from the CLI
    """

    if args.client_protocol == 'http':
        client = TritonClientFactory.create_http_client(
            server_url=args.triton_http_endpoint)
    elif args.client_protocol == 'grpc':
        client = TritonClientFactory.create_grpc_client(
            server_url=args.triton_grpc_endpoint)
    else:
        raise TritonModelAnalyzerException(
            f"Unrecognized client-protocol : {args.client_protocol}")

    return client


def get_server_handle(args):
    """
    Creates and returns a TritonServer
    with specified arguments

    Parameters
    ----------
    args : namespace
        Arguments parsed from the CLI
    """

    if args.triton_launch_mode == 'remote':
        triton_config = TritonServerConfig()
        triton_config['model-repository'] = 'remote-model-repository'
        logger.info('Using remote Triton Server...')
        server = TritonServerFactory.create_server_local(path=None,
                                                         config=triton_config)
    elif args.triton_launch_mode == 'local':
        triton_config = TritonServerConfig()
        triton_config['model-repository'] = args.model_repository
        triton_config['http-port'] = args.triton_http_endpoint.split(':')[-1]
        triton_config['grpc-port'] = args.triton_grpc_endpoint.split(':')[-1]
        triton_config['metrics-port'] = urlparse(args.triton_metrics_url).port
        triton_config['model-control-mode'] = 'explicit'
        logger.info('Starting a local Triton Server...')
        server = TritonServerFactory.create_server_local(
            path=args.triton_server_path, config=triton_config)
        server.start()
    elif args.triton_launch_mode == 'docker':
        triton_config = TritonServerConfig()
        triton_config['model-repository'] = args.model_repository
        triton_config['http-port'] = args.triton_http_endpoint.split(':')[-1]
        triton_config['grpc-port'] = args.triton_grpc_endpoint.split(':')[-1]
        triton_config['metrics-port'] = urlparse(args.triton_metrics_url).port
        triton_config['model-control-mode'] = 'explicit'
        logger.info('Starting a Triton Server using docker...')
        server = TritonServerFactory.create_server_docker(
            image='nvcr.io/nvidia/tritonserver:' + args.triton_version,
            config=triton_config,
            gpus=get_analyzer_gpus(args))
        server.start()
    else:
        raise TritonModelAnalyzerException(
            f"Unrecognized triton-launch-mode : {args.triton_launch_mode}")

    return server


def get_analyzer_gpus(args):
    """
    Creates a list of GPU UUIDs corresponding to the GPUs visible to
    model_analyzer.

    Parameters
    ----------
    args : namespace
        The arguments passed into the CLI
    """

    if len(args.gpus) == 1 and args.gpus[0] == 'all':
        devices = numba.cuda.list_devices()
    else:
        devices = args.gpus

    model_analyzer_gpus = []
    for device in devices:
        gpu_device = GPUDeviceFactory.create_device_by_cuda_index(device.id)
        model_analyzer_gpus.append(
            str(gpu_device.device_uuid(), encoding='ascii'))

    return model_analyzer_gpus


def get_triton_metrics_gpus(args):
    """
    Uses prometheus to request a list of GPU UUIDs corresponding to the GPUs
    visible to Triton Inference Server

    Parameters
    ----------
    args : namespace
        The arguments passed into the CLI
    """

    triton_prom_str = str(requests.get(args.triton_metrics_url).content,
                          encoding='ascii')
    metrics = text_string_to_metric_families(triton_prom_str)

    triton_gpus = []
    for metric in metrics:
        if metric.name == 'nv_gpu_utilization':
            for sample in metric.samples:
                triton_gpus.append(sample.labels['gpu_uuid'])

    return triton_gpus


def check_triton_and_model_analyzer_gpus(args):
    """
    Check whether Triton Server and Model Analyzer are using the same GPUs

    Parameters
    ----------
    args : namespace
        The arguments passed into the CLI

    Raises
    ------
    TritonModelAnalyzerException
        If they are using different GPUs this exception will be raised.
    """

    model_analyzer_gpus = get_analyzer_gpus(args)
    triton_gpus = get_triton_metrics_gpus(args)
    if set(model_analyzer_gpus) != set(triton_gpus):
        raise TritonModelAnalyzerException(
            "'Triton Server is not using the same GPUs as Model Analyzer: '"
            f"Model Analyzer GPUs {model_analyzer_gpus}, Triton GPUs {triton_gpus}"
        )


def get_triton_handles(args):
    """
    Creates a TritonServer and starts it. Creates a TritonClient

    Parameters
    ----------
    args : namespace
        The arguments passed into the CLI

    Returns
    -------
    TritonClient, TritonServer
        Handles for triton client/server pair.
    """

    client = get_client_handle(args)
    server = get_server_handle(args)
    client.wait_for_server_ready(num_retries=args.max_retries)
    logger.info('Triton Server is ready.')

    return client, server


def create_run_configs(args):
    """
    Parameters
    ----------
    args : namespace
        The arguments passed into the CLI

    Returns
    -------
    list of dicts
        keys are parameters to perf_analyzer
        values are individual combinations of argument values
    """

    sweep_params = {
        'model-name': [name.strip() for name in args.model_names.split(',')],
        'batch-size': [batch.strip() for batch in args.batch_sizes.split(',')],
        'concurrency-range': [c.strip() for c in args.concurrency.split(',')],
        'protocol': [args.client_protocol],
        'url': [
            args.triton_http_endpoint
            if args.client_protocol == 'http' else args.triton_grpc_endpoint
        ],
    }
    param_combinations = list(product(*tuple(sweep_params.values())))
    run_params = [
        dict(zip(sweep_params.keys(), vals)) for vals in param_combinations
    ]

    return run_params


def write_results(args, analyzer):
    """
    Makes calls to the analyzer to write results out to streams or files. If
    exporting results is requested, uses a FileWriter for specified output
    files.

    Parameters
    ----------
    args : namespace
        The arguments passed into the CLI
    analyzer : Analyzer
        The instance being used to profile
        server inferencing.
    """

    analyzer.write_results(writer=FileWriter(), column_separator=' ')
    if args.export:
        with open(os.path.join(args.export_path, args.filename_server_only),
                  'w+') as f:
            analyzer.export_server_only_csv(writer=FileWriter(file_handle=f),
                                            column_separator=',')
        with open(os.path.join(args.export_path, args.filename_model),
                  'w+') as f:
            analyzer.export_model_csv(writer=FileWriter(file_handle=f),
                                      column_separator=',')


def run_analyzer(args, analyzer, client, run_configs):
    """
    Makes a single call to profile the server only Then for each run
    configurations, it profiles model inference.

    Parameters
    ----------
    args : namespace
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
        client.wait_for_model_ready(model=model, num_retries=args.max_retries)
        try:
            analyzer.profile_model(run_config=run_config,
                                   perf_output_writer=FileWriter())
        finally:
            client.unload_model(model=model)


def main():
    """
    Main entrypoint of model_analyzer
    """

    global exiting

    # Number of Times User Requested Exit
    exiting = 0

    monitoring_metrics = [
        PerfThroughput, GPUUtilization, GPUUsedMemory, GPUFreeMemory
    ]

    def interrupt_handler(signal, frame):
        global exiting
        exiting += 1
        logging.info(
            f'Received SIGINT. Exiting ({exiting}/{MAX_NUMBER_OF_INTERRUPTS})...'
        )

        if exiting == MAX_NUMBER_OF_INTERRUPTS:
            sys.exit(1)
        return

    signal.signal(signal.SIGINT, interrupt_handler)

    try:
        args = CLI().parse()
    except TritonModelAnalyzerException as e:
        logging.error(f'Model Analyzer encountered an error: {e}')
        sys.exit(1)

    logging.info(f'Triton Model Analyzer started {args} arguments')
    analyzer = Analyzer(args, monitoring_metrics)
    server = None
    try:
        client, server = get_triton_handles(args)

        # Only check for exit after the events that take a long time.
        if exiting:
            return

        check_triton_and_model_analyzer_gpus(args)
        run_configs = create_run_configs(args)
        if exiting:
            return

        logging.info('Starting perf_analyzer...')
        run_analyzer(args, analyzer, client, run_configs)
        write_results(args, analyzer)
    except TritonModelAnalyzerException as e:
        logging.error(f'Model Analyzer encountered an error: {e}')
    finally:
        if server is not None:
            server.stop()


if __name__ == '__main__':
    main()
