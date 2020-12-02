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
import logging
from argparse import ArgumentParser

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

logger = logging.getLogger(__name__)


class CLI:
    """
    CLI class to parse the commandline arguments
    """
    def __init__(self):
        self._parser = ArgumentParser()
        self._add_arguments()

    def _add_arguments(self):
        # yapf:disable
        self._parser.add_argument(
            '-m',
            '--model-repository',
            type=str,
            required=True,
            help='Model repository location')
        self._parser.add_argument(
            '-n',
            '--model-names',
            type=str,
            required=True,
            help='Comma-delimited list of the model names to be profiled')
        self._parser.add_argument(
            '-b',
            '--batch-sizes',
            type=str,
            default='1',
            help='Comma-delimited list of batch sizes to use for the profiling')
        self._parser.add_argument(
            '-c',
            '--concurrency',
            type=str,
            default='1',
            help="Comma-delimited list of concurrency values or ranges <start:end:step>"
                 " to be used during profiling")
        self._parser.add_argument(
            '--export',
            action='store_true',
            help='Enables exporting metrics to a file')
        self._parser.add_argument(
            '-e',
            '--export-path',
            type=str,
            default='.',
            help='Full path to directory in which to store the results')
        self._parser.add_argument(
            '--filename-model',
            type=str,
            default='metrics-model.csv',
            help='Specifies filename for model running metrics')
        self._parser.add_argument(
            '--filename-server-only',
            type=str,
            default='metrics-server-only.csv',
            help='Specifies filename for server-only metrics')
        self._parser.add_argument(
            '-r',
            '--max-retries',
            type=int,
            default=100,
            help='Specifies the max number of retries for any retry attempt')
        self._parser.add_argument(
            '-d',
            '--duration-seconds',
            type=float,
            default=5,
            help='Specifies how long (seconds) to gather server-only metrics')
        self._parser.add_argument(
            '-i',
            '--monitoring-interval',
            type=float,
            default=0.01,
            help='Interval of time between DGCM measurements in seconds')
        self._parser.add_argument(
            '--client-protocol',
            type=str,
            choices=['http', 'grpc'],
            default='grpc',
            help='The protocol used to communicate with the Triton Inference Server')
        self._parser.add_argument(
            '--perf-analyzer-path',
            type=str,
            default='perf_analyzer',
            help='The full path to the perf_analyzer binary executable')
        self._parser.add_argument(
            '--triton-launch-mode',
            type=str,
            choices=['local', 'docker', 'remote'],
            default='local',
            help="The method by which to launch Triton Server. "
                 "'local' assumes tritonserver binary is available locally. "
                 "'docker' pulls and launches a triton docker container with "
                 "the specified version. 'remote' connects to a running "
                 "server using given http, grpc and metrics endpoints. "
        )
        self._parser.add_argument(
            '--triton-version',
            default='20.11-py3',
            type=str,
            help='Triton Server version')
        self._parser.add_argument(
            '--log-level',
            default='INFO',
            type=str,
            choices=['INFO', 'DEBUG', 'ERROR', 'WARNING'],
            help='Logging levels')
        self._parser.add_argument(
            '--triton-http-endpoint',
            type=str,
            default='localhost:8000',
            help="Triton Server HTTP endpoint url used by Model Analyzer client. "
                 "Will be ignored if server-launch-mode is not 'remote'")
        self._parser.add_argument(
            '--triton-grpc-endpoint',
            type=str,
            default='localhost:8001',
            help="Triton Server HTTP endpoint url used by Model Analyzer client. "
                 "Will be ignored if server-launch-mode is not 'remote'")
        self._parser.add_argument(
            '--triton-metrics-url',
            type=str,
            default='http://localhost:8002/metrics',
            help="Triton Server Metrics endpoint url. "
                 "Will be ignored if server-launch-mode is not 'remote'")
        self._parser.add_argument(
            '--triton-server-path',
            type=str,
            default='tritonserver',
            help='The full path to the tritonserver binary executable')
        self._parser.add_argument(
            '--gpus',
            type=str,
            default='all',
            help="List of GPU UUIDs to be used for the profiling. "
                 "Use 'all' to profile all the GPUs visible by CUDA.")
        # yapf:enable

    def _preprocess_and_verify_arguments(self, args):
        """
        Enforces some rules on input
        arguments. Sets some defaults.

        Parameters
        ----------
        args : argparse.Namespace
            containing all the parsed arguments

        Raises
        ------
        TritonModelAnalyzerException
            If arguments are passed in incorrectly
        """

        if args.export:
            if not args.export_path:
                logger.warning(
                    "--export-path specified without --export flag: skipping exporting metrics."
                )
                args.export_path = None
            elif args.export_path and not os.path.isdir(args.export_path):
                raise TritonModelAnalyzerException(
                    f"Export path {args.export_path} is not a directory.")
        if args.triton_launch_mode == 'remote':
            if args.client_protocol == 'http' and not args.triton_http_endpoint:
                raise TritonModelAnalyzerException(
                    "client-protocol is 'http'. Must specify triton-http-endpoint "
                    "if connecting to already running server or change protocol using "
                    "--client-protocol.")
            if args.client_protocol == 'grpc' and not args.triton_grpc_endpoint:
                raise TritonModelAnalyzerException(
                    "client-protocol is 'grpc'. Must specify triton-grpc-endpoint "
                    "if connecting to already running server or change protocol using "
                    "--client-protocol.")
        args.gpus = args.gpus.split(',')

    def _setup_logger(self, args):
        """
        Setup logger format
        """
        log_level = logging.getLevelName(args.log_level)
        logging.basicConfig(level=log_level,
                            format="%(asctime)s.%(msecs)d %(levelname)-4s"
                            "[%(filename)s:%(lineno)d] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    def parse(self):
        """
        Retrieves the arguments from the command line and loads them into an
        ArgumentParser Also does some sanity checks for arguments.

        Returns
        -------
        argparse.Namespace
            containing all the parsed arguments

        Raises
        ------
        TritonModelAnalyzerException
            For arguments passed incorrectly
        """

        # Remove the first argument which is the program name
        args = self._parser.parse_args(sys.argv[1:])
        self._preprocess_and_verify_arguments(args)
        self._setup_logger(args)

        return args
