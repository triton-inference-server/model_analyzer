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

from argparse import ArgumentParser


class CLI:
    """
    CLI class to parse the commandline arguments
    """

    def __init__(self):
        self._parser = ArgumentParser(prog='model_analyzer')
        self._add_arguments()

    def _add_arguments(self):
        self._parser.add_argument(
            '-m',
            '--model-repository',
            type=str,
            required=True,
            help='Model repository location')
        self._parser.add_argument(
            '-t',
            '--triton-version',
            type=str,
            help='Triton Server version')
        self._parser.add_argument(
            '-n',
            '--model-names',
            type=str,
            required=True,
            help='Comma delimited list of the model names to be profiled')
        self._parser.add_argument(
            '-b',
            '--batch-size',
            type=int,
            default=1,
            help='Batch size to use for the profiling')
        self._parser.add_argument(
            '-e',
            '--export',
            type=str,
            help='Export path to store the results')
        self._parser.add_argument(
            '-c',
            '--concurrency-range',
            type=str,
            help='Concurrency range values <start:end:step>')

    def parse(self, argv):
        # Remove the first argument which is the program name
        return self._parser.parse_args(argv[1:])
