# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
from itertools import product

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
        server = TritonServerFactory.create_server_local(path=None,
                                                         config=triton_config)
    elif args.triton_launch_mode == 'local':
        triton_config = TritonServerConfig()
        triton_config['model-repository'] = args.model_repository
        triton_config['model-control-mode'] = 'explicit'
        server = TritonServerFactory.create_server_local(
            path=args.triton_server_path, config=triton_config)
        server.start()
    elif args.triton_launch_mode == 'docker':
        triton_config = TritonServerConfig()
        triton_config['model-repository'] = args.model_repository
        triton_config['model-control-mode'] = 'explicit'
        server = TritonServerFactory.create_server_docker(
            model_path=args.model_repository,
            image='nvcr.io/nvidia/tritonserver:' + args.triton_version,
            config=triton_config)
        server.start()
    else:
        raise TritonModelAnalyzerException(
            f"Unrecognized triton-launch-mode : {args.triton_launch_mode}")

    return server


def get_triton_handles(args):
    """
    Creates a TritonServer and starts it.
    Creates a TritonClient
    
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
        'model-name': args.model_names.split(','),
        'batch-size': args.batch_sizes.split(','),
        'concurrency-range': args.concurrency.split(',')
    }
    param_combinations = list(product(*tuple(sweep_params.values())))
    run_params = [
        dict(zip(sweep_params.keys(), vals)) for vals in param_combinations
    ]

    return run_params


def write_results(args, analyzer):
    """
    Makes calls to the analyzer to write
    results out to streams or files.
    If exporting results is requested,
    uses a FileWriter for specified output
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
                                            column_separator=',',
                                            ignore_widths=True)
        with open(os.path.join(args.export_path, args.filename_model),
                  'w+') as f:
            analyzer.export_model_csv(writer=FileWriter(file_handle=f),
                                      column_separator=',',
                                      ignore_widths=True)


def run_analyzer(args, analyzer, client, run_configs):
    """
    Makes a single call to profile the server only
    Then for each run configurations, it profiles
    model inference.

    Parameters
    ----------
    args : namespace
        The arguments passed into the CLI
    analyzer : Analyzer
        The instance being used to profile
        server inferencing.
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

    args = CLI().parse()
    monitoring_metrics = [
        PerfThroughput, GPUUtilization, GPUUsedMemory, GPUFreeMemory
    ]

    analyzer = Analyzer(args, monitoring_metrics)

    client, server = get_triton_handles(args)
    run_configs = create_run_configs(args)

    try:
        run_analyzer(args, analyzer, client, run_configs)
    finally:
        server.stop()

    write_results(args, analyzer)


if __name__ == '__main__':
    main()
