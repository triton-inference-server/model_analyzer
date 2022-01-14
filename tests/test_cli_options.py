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

import os
import subprocess

from model_analyzer.config.input.config_defaults import DEFAULT_TRITON_DOCKER_IMAGE

from .common import test_result_collector as trc

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_analyze import ConfigCommandAnalyze
from model_analyzer.config.input.config_command_report import ConfigCommandReport
from model_analyzer.config.input.config_status import ConfigStatus
from model_analyzer.constants import CONFIG_PARSER_SUCCESS
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

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
@patch(
    'model_analyzer.config.input.config_command_profile.binary_path_validator',
    lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS))
class TestCLIOptions(trc.TestResultCollector):
    """
    Tests the methods of the CLI class
    """

    def test_basic_cli_config_options(self):
        """
        Test the minimal set of cli commands necessary to run Model Analyzer profile
        """
        cli = CLIConfigStruct()
        args, config = cli.parse()
        # print(f"NUMBER OF ARGS: {len(vars(args))}")
        # print(f"ARGS: {vars(args)}")
        # print(f"config: {config.get_config().keys()}")
        # print(f"number of config: {len(config.get_config().keys())}")
        model_repo = config.model_repository
        profile_model = config.profile_models[0].model_name()
        self.assertEqual('foo', model_repo)
        self.assertEqual('bar', profile_model)

    def test_boolean_options(self):
        #yapf: disable
        options = [
            "--override-output-model-repository",
            "--use-local-gpu-monitor",
            "--collect-cpu-metrics",
            "--perf-output",
            "--run-config-search-disable"
        ]
        #yapf: enable
        for option in options:
            self._test_boolean_option(option)

    # @patch.object(ConfigCommandProfile, '_load_config_file')
    @patch(
        'model_analyzer.config.input.config_command_profile.ConfigCommandProfile._load_config_file'
    )
    def test_string_options(self, mocked_load_config_file):
        #yapf: disable
        # Options format:
        #   (long_flag, short_flag, test_value, default_value, expected_failing_value)
        # The following options can be None:
        #   short_flag
        #   default_value
        #   expected_failing_value
        #TODO: ask Tim if launch mode, client protocol are handled correctly
        options = [
            ("--config-file", "-f", "baz", None, None),
            ("--checkpoint-directory", "-s", "./test_dir", os.path.join(os.getcwd(), "checkpoints"), None),
            ("--output-model-repository-path", None, "./test_dir", os.path.join(os.getcwd(), "output_model_repository"), None),
            ("--client-protocol", None, "http", "grpc", "SHOULD_FAIL"),
            ("--client-protocol", None, "grpc", "grpc", "SHOULD_FAIL"),
            ("--perf-analyzer-path", None, ".", "perf_analyzer", None),
            ("--perf-output-path", None, ".", None, None),
            ("--triton-docker-image", None, "test_image", DEFAULT_TRITON_DOCKER_IMAGE, None),
            ("--triton-http-endpoint", None, "localhost:4000", "localhost:8000", None),
            ("--triton-grpc-endpoint", None, "localhost:4001", "localhost:8001", None),
            ("--triton-metrics-url", None, "localhost:4002", "http://localhost:8002/metrics", None),
            ("--triton-server-path", None, "test_path", "tritonserver", None),
            ("--triton-output-path", None, "test_path", None, None),
            ("--triton-launch-mode", None, "local", "local", "SHOULD_FAIL"),
            ("--triton-launch-mode", None, "docker", "local", None),
            ("--triton-launch-mode", None, "remote", "local", None),
            ("--triton-launch-mode", None, "c_api", "local", None)
        ]
        #yapf: enable
        for option_tuple in options:
            self._test_string_option(option_tuple)

    def test_int_options(self):
        #yapf: disable
        # Options format:
        #   (long_option, short_option, test_value, default_value)
        # The following options can be None:
        #   short_option
        #   default_value

        options = [
            ("--client-max-retries", "-r", "125", "50"),
            ("--duration-seconds", "-d", "10", "3"),
            ("--perf-analyzer-timeout", None, "100", "600"),
            ("--run-config-search-max-concurrency", None, "100", "1024"),
            ("--run-config-search-max-instance-count", None, "10", "5")
        ]
        #yapf: enable
        for option_tuple in options:
            self._test_int_option(option_tuple)

    def _test_boolean_option(self, option):
        option_with_underscores = self._convert_flag(option)
        # print(f"\n>>> {option}")
        cli = CLIConfigStruct()
        _, config = cli.parse()
        option_value = config.get_config().get(option_with_underscores).value()
        self.assertEqual(option_value, False)

        cli = CLIConfigStruct()
        cli.args.extend([option])
        _, config = cli.parse()
        option_value = config.get_config().get(option_with_underscores).value()
        self.assertEqual(option_value, True)

        cli = CLIConfigStruct()
        cli.args.extend([option, 'SHOULD_FAIL'])
        with self.assertRaises(SystemExit):
            _, config = cli.parse()

    def _test_string_option(self, option_tuple):
        long_option = option_tuple[0]
        short_option = option_tuple[1]
        expected_value = option_tuple[2]
        default_value = option_tuple[3]
        expected_failing_value = option_tuple[4]

        # print(
        # f"\n>>> {long_option},{short_option}, {expected_value}, {default_value}, {expected_failing_value}"
        # )
        long_option_with_underscores = self._convert_flag(long_option)

        # print(f"\t>>> long option flag: {long_option}, {expected_value}")
        cli = CLIConfigStruct()
        cli.args.extend([long_option, expected_value])
        _, config = cli.parse()
        option_value = config.get_config().get(
            long_option_with_underscores).value()
        self.assertEqual(option_value, expected_value)

        if short_option is not None:
            # print(f"\t>>> short option flag: {short_option}, {expected_value}")
            cli = CLIConfigStruct()
            cli.args.extend([short_option, expected_value])
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).value()
            self.assertEqual(option_value, expected_value)

        if default_value is not None:
            # print(f"\t>>> default value: {long_option}, {default_value}")
            cli = CLIConfigStruct()
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).default_value()
            self.assertEqual(option_value, default_value)

        if expected_failing_value is not None:
            # print(f"\t>>> error value: {long_option}, {expected_failing_value}")
            cli = CLIConfigStruct()
            cli.args.extend([long_option, expected_failing_value])
            with self.assertRaises(SystemExit):
                _, config = cli.parse()

    def _test_int_option(self, option_tuple):
        long_option = option_tuple[0]
        short_option = option_tuple[1]
        expected_value = option_tuple[2]
        default_value = option_tuple[3]

        print(
            f"\n>>> {long_option},{short_option}, {expected_value}, {default_value}"
        )
        long_option_with_underscores = self._convert_flag(long_option)

        print(f"\t>>> long option flag: {long_option}, {expected_value}")
        cli = CLIConfigStruct()
        cli.args.extend([long_option, expected_value])
        _, config = cli.parse()
        option_value = config.get_config().get(
            long_option_with_underscores).value()
        self.assertEqual(option_value, int(expected_value))

        if short_option is not None:
            print(f"\t>>> short option flag: {short_option}, {expected_value}")
            cli = CLIConfigStruct()
            cli.args.extend([short_option, expected_value])
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).value()
            self.assertEqual(option_value, int(expected_value))

        if default_value is not None:
            print(f"\t>>> default value: {long_option}, {default_value}")
            cli = CLIConfigStruct()
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).default_value()
            self.assertEqual(option_value, int(default_value))

    def _convert_flag(self, option):
        return option.lstrip("-").replace("-", "_")