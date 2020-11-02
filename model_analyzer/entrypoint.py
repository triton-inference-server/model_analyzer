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
import time
from itertools import product

from .cli.cli import CLI

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from .monitor.nvml import NVMLMonitor
from .perf_analyzer.perf_analyzer import PerfAnalyzer
from .perf_analyzer.perf_config import PerfAnalyzerConfig
from .triton.server.server_factory import TritonServerFactory
from .triton.server.server_config import TritonServerConfig
from .triton.client.client_factory import TritonClientFactory
from .triton.client.client_config import TritonClientConfig
from .triton.model.model import Model
from .record.record_aggregator import RecordAggregator
from .record.gpu_free_memory import GPUFreeMemory
from .record.gpu_used_memory import GPUUsedMemory
from .record.perf_throughput import PerfThroughput
from .output.output_table import OutputTable
from .output.file_writer import FileWriter


def create_perf_config(params):
    """
    Utility function for creating
    a PerfAnalyzerConfig from
    a dict of parameters

    Parameters
    ----------
    params : dict
        keys are arguments to perf analyzer
        values are their value

    Returns
    -------
    PerfAnalyzerConfig
        The arguments from params
        are set in this config
    """

    config = PerfAnalyzerConfig()
    for param, value in params.items():
        config[param] = value
    return config


def main():
    """
    Main entrypoint of model_analyzer
    """

    # Get args
    cli = CLI()
    args = cli.parse(sys.argv)
    print(args)

    # Create and start tritonserver
    triton_config = TritonServerConfig()
    triton_config['model-repository'] = args.model_repository
    triton_config['model-control-mode'] = 'explicit'

    server = TritonServerFactory.create_server_local(config=triton_config)
    server.start()
    server.wait_for_ready()

    # Create triton client and model
    client_config = TritonClientConfig()
    if args.client_protocol == 'http':
        client_config['url'] = 'localhost:8000'
        client = TritonClientFactory.create_http_client(config=client_config)
    elif args.client_protocol == 'grpc':
        client_config['url'] = 'localhost:8001'
        client = TritonClientFactory.create_grpc_client(config=client_config)

    # To run perf_analyzer we need all combinations of configs
    start, stop, step = tuple(map(int, args.concurrency_range.split(':')))
    sweep_params = {
        'model-name': args.model_names.split(','),
        'batch-size': args.batch_size.split(','),
        'concurrency-range': list(range(start, stop + 1, step))
    }

    param_combinations = list(product(*tuple(sweep_params.values())))
    run_params = [
        dict(zip(sweep_params.keys(), vals)) for vals in param_combinations
    ]

    # Create a record aggregator to collect records, and output table
    record_aggregator = RecordAggregator()
    table_headers = [
        "Model", "Batch", "Concurrency",
        PerfThroughput.header(), "Max " + GPUFreeMemory.header(),
        "Max " + GPUUsedMemory.header()
    ]

    # Create the server only result
    server_only_table = OutputTable(headers=table_headers)
    nvml_monitor = NVMLMonitor(frequency=args.monitoring_interval)
    nvml_monitor.start_recording_metrics(["memory"])
    time.sleep(1)
    nvml_records = nvml_monitor.stop_recording_metrics()

    # Insert all records into aggregator
    for record in nvml_records:
        record_aggregator.insert(record)

    # Get max GPUUsedMemory and GPUFreeMemory
    aggregated_records = record_aggregator.aggregate(
        headers=[GPUUsedMemory.header(),
                 GPUFreeMemory.header()],
        reduce_func=max)

    server_only_table.add_row([
        "triton-server", 0, 0, 0, aggregated_records[GPUFreeMemory.header()],
        aggregated_records[GPUUsedMemory.header()]
    ])

    record_aggregator = RecordAggregator()

    # Now create the output table for the model runs
    model_table = OutputTable(headers=table_headers)

    # Create table writer for stdout
    writer = FileWriter()

    # For each combination of run parameters get measurements
    for params in run_params:
        perf_config = create_perf_config(params)
        model = Model(perf_config['model-name'])

        # load the model
        client.load_model(model=model)

        # Configure perf_analyzer and monitors
        perf_analyzer = PerfAnalyzer(config=perf_config)

        # Start monitors and run perf_analyzer
        nvml_monitor.start_recording_metrics(["memory"])
        throughput_record, _ = perf_analyzer.run()
        nvml_records = nvml_monitor.stop_recording_metrics()
        writer.write(perf_analyzer.output())

        # Insert all records into aggregator
        for record in nvml_records:
            record_aggregator.insert(record)

        # Get max GPUUsedMemory and GPUFreeMemory
        aggregated_records = record_aggregator.aggregate(
            headers=[GPUUsedMemory.header(),
                     GPUFreeMemory.header()],
            reduce_func=max)

        # Create row from records
        output_row = []
        output_row.append(model.name())
        output_row.append(perf_config['batch-size'])
        output_row.append(perf_config['concurrency-range'])
        output_row.append(throughput_record.value())
        output_row.append(aggregated_records[GPUFreeMemory.header()])
        output_row.append(aggregated_records[GPUUsedMemory.header()])

        # Add row and then unload model
        model_table.add_row(output_row)
        client.unload_model(model=model)

    # Stop triton
    server.stop()
    nvml_monitor.destroy()

    # Write output
    writer.write("Server Only:")
    writer.write(
        server_only_table.to_formatted_string(column_width=28, separator=' ') +
        '\n')
    writer.write("Models:")
    writer.write(
        model_table.to_formatted_string(column_width=28, separator=' '))


if __name__ == '__main__':
    main()
