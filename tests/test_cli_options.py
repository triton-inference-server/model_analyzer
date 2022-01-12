# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

from .common import test_result_collector as trc

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_analyze import ConfigCommandAnalyze
from model_analyzer.config.input.config_command_report import ConfigCommandReport
from model_analyzer.constants import CONFIG_PARSER_FAILURE, CONFIG_PARSER_SUCCESS
from model_analyzer.config.input.config_status import ConfigStatus

from unittest.mock import patch


class CLISubclass(CLI):
    """
    Subclass of CLI to overwrite the parse method.
    Parse takes a list of arguments instead of getting the args
    from sys.argv
    """

    def __init__(self):
        super().__init__()

    def parse(self, parsed_commands=None):
        args = self._parser.parse_args(parsed_commands[1:])
        if args.subcommand is None:
            self._parser.print_help()
            self._parser.exit()
        config = self._subcommand_configs[args.subcommand]
        config.set_config_values(args)
        return args, config


class CLIConfigStruct():
    """
    Struct class to hold the common variables shared between tests
    """

    def __init__(self):
        #yapf: disable
        self.args = [
            '/usr/local/bin/model-analyzer',
            'profile',
            '--model-repository',
            'foo',
            '--profile-models',
            'bar'
        ]
        #yapf: enable
        config_profile = ConfigCommandProfile()
        self.cli = CLISubclass()
        self.cli.add_subcommand(cmd='profile', help='', config=config_profile)

    def parse(self):
        return self.cli.parse(self.args)


@patch('model_analyzer.config.input.config_command_profile.file_path_validator',
       lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS))
class TestCLIOptions(trc.TestResultCollector):
    """
    Tests the methods of the CLI class
    """

    # @patch(
    #     'model_analyzer.config.input.config_command_profile.file_path_validator',
    #     lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS))
    def test_basic_cli_config(self):
        """
        Test the minimal set of cli commands necessary to run Model Analyzer profile
        """
        # #yapf: disable
        # sys.argv = [
        #     '/usr/local/bin/model-analyzer',
        #     'profile',
        #     '--model-repository',
        #     'foo',
        #     '--profile-models',
        #     'bar'
        # ]
        # #yapf: enable
        # config_profile = ConfigCommandProfile()
        # cli = CLI()
        # cli.add_subcommand(cmd='profile', help='', config=config_profile)
        cli = CLIConfigStruct()
        _, config = cli.parse()
        model_repo = config.model_repository
        profile_model = config.profile_models[0].model_name()
        self.assertEqual('foo', model_repo)
        self.assertEqual('bar', profile_model)

    # @patch.object(ConfigCommandProfile, '_load_config_file')
    # @patch(
    #     'model_analyzer.config.input.config_command_profile.file_path_validator',
    #     lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS))




    @patch(
        'model_analyzer.config.input.config_command_profile.ConfigCommandProfile._load_config_file'
    )
    def test_config_file_short_option(self, mocked_load_config_file):
        """
        Test the -f flag
        """
        # #yapf: disable
        # sys.argv = [
        #     '/usr/local/bin/model-analyzer',
        #     'profile',
        #     '--model-repository',
        #     'foo',
        #     '--profile-models',
        #     'bar'
        # ]
        # #yapf: enable
        cli = CLIConfigStruct()
        test_file_name = 'baz'
        cli.args.extend(['-f', test_file_name])
        # sys.argv.extend(['-f', test_file_name])
        # config_profile = ConfigCommandProfile()
        # cli = CLI()
        # cli.add_subcommand(cmd='profile', help='', config=config_profile)
        _, config = cli.parse()
        config_file_name = config.config_file
        self.assertEqual(config_file_name, test_file_name)