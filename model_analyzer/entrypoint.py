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
from .triton.server.server_factory import TritonServerFactory
from .triton.server.server_config import TritonServerConfig
from .triton.client.client_factory import TritonClientFactory
from .triton.model.model import Model
from .record.gpu_free_memory import GPUFreeMemory
from .record.gpu_used_memory import GPUUsedMemory
from .record.perf_throughput import PerfThroughput
from .record.perf_latency import PerfLatency
from .output.file_writer import FileWriter


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

    triton_config = TritonServerConfig()
    triton_config['model-repository'] = args.model_repository
    triton_config['model-control-mode'] = 'explicit'
    server = TritonServerFactory.create_server_local(
        path=args.triton_server_path, config=triton_config)
    if args.client_protocol == 'http':
        client = TritonClientFactory.create_http_client(
            server_url='localhost:8000')
    elif args.client_protocol == 'grpc':
        client = TritonClientFactory.create_grpc_client(
            server_url='localhost:8001')

    server.start()
    server.wait_for_ready(num_retries=args.max_retries)

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


def main():
    """
    Main entrypoint of model_analyzer
    """

    # Get args and set monitoring metrics
    args = CLI().parse()
    monitoring_metrics = [PerfThroughput, GPUUsedMemory, GPUFreeMemory]

    # Triton handles and analyzer
    client, server = get_triton_handles(args)
    analyzer = Analyzer(args=args, monitoring_metrics=monitoring_metrics)

    # To run perf_analyzer we need all combinations of configs
    run_configs = create_run_configs(args)

    try:
        # Server only metrics
        analyzer.profile_server_only()

        # Model inference metrics
        for run_config in run_configs:
            model = Model(name=run_config['model-name'])
            client.load_model(model=model)
            client.wait_for_model_ready(model=model,
                                        num_retries=args.max_retries)
            analyzer.profile_model(run_config=run_config,
                                   perf_output_writer=FileWriter())
            client.unload_model(model=model)

    finally:
        server.stop()

    # Write output tables
    analyzer.write_server_only_result(writer=FileWriter(),
                                      column_width=28,
                                      column_separator=' ')
    analyzer.write_model_result(writer=FileWriter(),
                                column_width=28,
                                column_separator=' ')
    if args.export:
        with open(os.join(args.export_path, args.filename_server_only),
                  'r+') as f:
            analyzer.write_server_only_result(writer=FileWriter(file_handle=f),
                                              column_width=None,
                                              column_separator=',')
        with open(os.join(args.export_path, args.filename_model), 'r+') as f:
            analyzer.write_model_result(writer=FileWriter(file_handle=f),
                                        column_width=None,
                                        column_separator=',')


if __name__ == '__main__':
    main()
