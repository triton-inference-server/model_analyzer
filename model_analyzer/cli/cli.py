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

import sys
import logging
import argparse
from argparse import ArgumentParser

logger = logging.getLogger(__name__)


class CLI:
    """
    CLI class to parse the commandline arguments
    """
    def __init__(self, config):
        self._parser = ArgumentParser()
        self._config = config
        self._add_arguments(config)

    def _add_arguments(self, config):
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
                    parser_args['action'] == 'store_true'
                    or parser_args['action'] == 'store_false'):
                self._parser.add_argument(
                    *config.flags(),
                    default=argparse.SUPPRESS,
                    help=config.description(),
                    **config.parser_args(),
                )
            else:
                self._parser.add_argument(
                    *config.flags(),
                    default=argparse.SUPPRESS,
                    choices=config.choices(),
                    help=config.description(),
                    type=config.cli_type(),
                    **config.parser_args(),
                )

    def parse(self):
        """
        Retrieves the arguments from the command line and loads them into an
        ArgumentParser. It will also configure the Model Analyzer config
        accordingly.
        """

        # Remove the first argument which is the program name
        args = self._parser.parse_args(sys.argv[1:])
        self._config.set_config_values(args)
