# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, Union, Optional, Tuple

import sys
import importlib_metadata
import logging
import argparse
from argparse import ArgumentParser, Namespace

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_report import ConfigCommandReport
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

from model_analyzer.constants import LOGGER_NAME, PACKAGE_NAME

logger = logging.getLogger(LOGGER_NAME)


class CLI:
    """
    CLI class to parse the command line arguments
    """

    def __init__(self):
        self._parser = ArgumentParser()
        self._add_global_options()
        self._subparsers = self._parser.add_subparsers(
            help='Subcommands under Model Analyzer', dest='subcommand')

        # Store subcommands, and their configs
        self._subcommand_configs = {}

    def _add_global_options(self):
        """
        Adds the Model Analyzer's global options
        to the parser
        """

        self._parser.add_argument(
            '-q',
            '--quiet',
            action='store_true',
            help='Suppress all output except for error messages.')
        self._parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            help='Show detailed logs, messages and status.')
        self._parser.add_argument(
            '-m',
            '--mode',
            type=str,
            default='online',
            choices=['online', 'offline'],
            help='Choose a preset configuration mode.')
        self._parser.add_argument(
            '--version',
            action='store_true',
            help='Show the Model Analyzer version.')

    def add_subcommand(self, cmd, help, config=None):
        """
        Adds a subparser to the main parser representing
        a command. Also adds the passed in config to
        the subcommands dict to set its values upon parse.

        Parameters
        ----------
        cmd : str
            subcommand name
        help: str
            help string or description for the subcommand
        config: Config
            The config containing the arguments that are required
            to be parsed for this subcommand.
        """

        subparser = self._subparsers.add_parser(cmd, help=help)
        if config:
            self._add_config_arguments(subparser, config)
            self._subcommand_configs[cmd] = config

    def _add_config_arguments(self, subparser, config):
        """
        Add the CLI arguments from the config

        Parameters
        ----------
        config : Config
            Model Analyzer config object.
        """
        #configs is dictionary of config_fields objects from config_command_*
        configs = config.get_config()
        for config_field in configs.values():
            parser_args = config_field.parser_args()

            # Skip the non-CLI flags
            if config_field.flags() is None:
                continue

            # 'store_true' and 'store_false' does not
            # allow 'type' or 'choices' parameters
            if 'action' in parser_args and (
                    parser_args['action'] == 'store_true' or
                    parser_args['action'] == 'store_false'):
                subparser.add_argument(
                    *config_field.flags(),
                    default=argparse.SUPPRESS,
                    help=config_field.description(),
                    **config_field.parser_args(),
                )
            else:
                subparser.add_argument(
                    *config_field.flags(),
                    default=argparse.SUPPRESS,
                    choices=config_field.choices(),
                    help=config_field.description(),
                    type=config_field.cli_type(),
                    **config_field.parser_args(),
                )

    def _show_model_analyzer_version(self):
        """
        Displays the current version of Model Analyzer and exits.
        """
        try:
            version = importlib_metadata.version(PACKAGE_NAME)
            print(version)
            sys.exit(0)
        except importlib_metadata.PackageNotFoundError:
            raise TritonModelAnalyzerException(
                f"Version information is not available")

    def parse(
        self,
        input_args: Optional[List] = None
    ) -> Tuple[Namespace, Union[ConfigCommandProfile, ConfigCommandReport]]:
        """
        Parse CLI options using ArgumentParsers 
        and set config values.

        Parameters
        ----------
        input_args: List
            The list of arguments to be parsed 
            (if None then command line arguments will be used)

        Returns
        -------
        args : Namespace
            Object that contains the parse CLI commands
            Used for the global options
        config: CommandConfig
            The config corresponding to the command being run,
            already filled in with values from CLI or YAML.
        """

        args = self._parser.parse_args(input_args)

        if args.version:
            self._show_model_analyzer_version()

        if args.subcommand is None:
            self._parser.print_help()
            self._parser.exit()
        config = self._subcommand_configs[args.subcommand]
        config.set_config_values(args)
        return args, config
