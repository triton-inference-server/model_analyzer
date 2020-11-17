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
from argparse import ArgumentParser

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


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
            '--base-duration',
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
                 "'docker' pulls and launches a triton docker container with the specified version. "
                 "'remote' connects to a running server using given http, grpc and metrics endpoints. "
        )
        self._parser.add_argument(
            '-v',
            '--triton-version',
            type=str,
            help='Triton Server version')
        self._parser.add_argument(
            '--triton-http-endpoint',
            type=str,
            help="Triton Server HTTP endpoint url used by Model Analyzer client. "
                 "Will be ignored if server-launch-mode is not 'remote'")
        self._parser.add_argument(
            '--triton-grpc-endpoint',
            type=str,
            help="Triton Server HTTP endpoint url used by Model Analyzer client. "
                 "Will be ignored if server-launch-mode is not 'remote'")
        self._parser.add_argument(
            '--triton-server-path',
            type=str,
            default='/opt/tritonserver/bin/tritonserver',
            help='The full path to the tritonserver binary executable')
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
                print(
                    "--export-path specified without --export flag: skipping exporting metrics."
                )
                args.export_path = None
            elif args.export_path and not os.path.isdir(args.export_path):
                raise TritonModelAnalyzerException(
                    f"Export path {args.export_path} is not a directory.")
        if args.triton_launch_mode != 'remote':
            if args.triton_http_endpoint or args.triton_grpc_endpoint:
                print(f"triton-launch-mode is {args.triton_launch_mode}."
                      " Specified Triton endpoints will be ignored.")
            args.triton_http_endpoint = 'localhost:8000'
            args.triton_grpc_endpoint = 'localhost:8001'
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
        elif args.triton_launch_mode == 'docker':
            if not args.triton_version:
                raise TritonModelAnalyzerException(
                    "triton-launch-mode is 'docker'. Must specify triton-version "
                    "if launching server docker container or change launch mode using "
                    "--triton-launch-mode.")

    def parse(self):
        """
        Retrieves the arguments from the command
        line and loads them into an ArgumentParser
        Also does some sanity checks for arguments.
        
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
        return args
