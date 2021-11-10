# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from argparse import ArgumentParser

from .common import test_result_collector as trc

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_analyze import ConfigCommandAnalyze
from model_analyzer.config.input.config_command_report import ConfigCommandReport
from model_analyzer.cli.cli import CLI


class ArgumentParserSubclass(ArgumentParser):

    def print_help(self, file=None):
        super().print_help(file)
        self._did_print_help = True

    _did_print_help = False


class CLISubclass(CLI):

    def __init__(self):
        self._parser = ArgumentParserSubclass()
        self._add_global_options()
        self._subparsers = self._parser.add_subparsers(
            help='Subcommands under Model Analyzer', dest='subcommand')

        # Store subcommands, and their configs
        self._subcommand_configs = {}


class TestCLI(trc.TestResultCollector):

    def test_help_message_no_args(self):
        sys.argv = ['/usr/local/bin/model-analyzer']

        config_profile = ConfigCommandProfile()
        config_analyze = ConfigCommandAnalyze()
        config_report = ConfigCommandReport()

        cli = CLISubclass()
        cli.add_subcommand(
            cmd='profile',
            help=
            'Run model inference profiling based on specified CLI or config options.',
            config=config_profile)
        cli.add_subcommand(
            cmd='analyze',
            help=
            'Collect and sort profiling results and generate data and summaries.',
            config=config_analyze)
        cli.add_subcommand(cmd='report',
                           help='Generate detailed reports for a single config',
                           config=config_report)

        self.assertRaises(SystemExit, cli.parse)
        self.assertTrue(cli._parser._did_print_help == True)
