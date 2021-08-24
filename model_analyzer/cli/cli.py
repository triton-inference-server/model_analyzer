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

import logging
import argparse
from argparse import ArgumentParser
from model_analyzer.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class CLI:
    """
    CLI class to parse the commandline arguments
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
            help='Show detailed logs, messags and status.')
        self._parser.add_argument('-m',
                                  '--mode',
                                  type=str,
                                  default='online',
                                  choices=['online', 'offline'],
                                  help='Choose a preset configuration mode.')

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

        configs = config.get_config()
        for config in configs.values():
            parser_args = config.parser_args()

            # Skip the non-CLI flags
            if config.flags() is None:
                continue

            # 'store_true' and 'store_false' does not
            # allow 'type' or 'choices' parameters
            if 'action' in parser_args and (
                    parser_args['action'] == 'store_true' or
                    parser_args['action'] == 'store_false'):
                subparser.add_argument(
                    *config.flags(),
                    default=argparse.SUPPRESS,
                    help=config.description(),
                    **config.parser_args(),
                )
            else:
                subparser.add_argument(
                    *config.flags(),
                    default=argparse.SUPPRESS,
                    choices=config.choices(),
                    help=config.description(),
                    type=config.cli_type(),
                    **config.parser_args(),
                )

    def parse(self):
        """
        Parse CLI options using ArgumentParsers 
        and set config values.

        Returns
        -------
        args : Namespace
            Object that contains the parse CLI commands
            Used for the global options
        config: CommandConfig
            The config corresponding to the command being run,
            already filled in with values from CLI or YAML.
        """

        args = self._parser.parse_args()
        config = self._subcommand_configs[args.subcommand]
        config.set_config_values(args)
        return args, config
