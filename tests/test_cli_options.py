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
import copy

from model_analyzer.config.input.config_defaults import DEFAULT_TRITON_DOCKER_IMAGE

from .common import test_result_collector as trc

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_analyze import ConfigCommandAnalyze
from model_analyzer.config.input.config_command_report import ConfigCommandReport
from model_analyzer.config.input.config_status import ConfigStatus
from model_analyzer.constants import CONFIG_PARSER_SUCCESS

import psutil

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


class OptionStruct():

    def __init__(self,
                 type_str,
                 stage,
                 long_flag,
                 short_flag=None,
                 expected_value=None,
                 default_value=None,
                 expected_failing_value=None,
                 extra_commands=None):

        self.long_flag = long_flag
        self.short_flag = short_flag
        self.expected_value = expected_value
        self.default_value = default_value
        self.expected_failing_value = expected_failing_value
        self.type = type_str
        self.extra_commands = extra_commands

        if stage == "profile":
            self.cli_subcommand = CLIConfigStruct


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

    @patch(
        'model_analyzer.config.input.config_command_profile.ConfigCommandProfile._load_config_file'
    )
    def test_all_options(self, mocked_load_config_file):

        #yapf: disable
        options = [
            #Boolean options
            # Options format:
            #   (bool, MA step, long_option)
            OptionStruct("bool", "profile","--override-output-model-repository"),
            OptionStruct("bool", "profile","--use-local-gpu-monitor"),
            OptionStruct("bool", "profile","--collect-cpu-metrics"),
            OptionStruct("bool", "profile","--perf-output"),
            OptionStruct("bool", "profile","--run-config-search-disable"),

            #Int/Float options
            # Options format:
            #   (int/float, MA step, long_option, short_option, test_value, default_value)
            # The following options can be None:
            #   short_option
            #   default_value
            OptionStruct("int", "profile", "--client-max-retries", "-r", "125", "50"),
            OptionStruct("int", "profile", "--duration-seconds", "-d", "10", "3"),
            OptionStruct("int", "profile", "--perf-analyzer-timeout", None, "100", "600"),
            OptionStruct("int", "profile", "--run-config-search-max-concurrency", None, "100", "1024"),
            OptionStruct("int", "profile", "--run-config-search-max-instance-count", None, "10", "5"),
            OptionStruct("float", "profile", "--monitoring-interval", "-i", "10.0", "1.0"),
            OptionStruct("float", "profile", "--perf-analyzer-cpu-util", None, "10.0", str(psutil.cpu_count() * 80.0)),

            #String options
            # Options format:
            #   (string, MA step, long_flag, short_flag, test_value, default_value, expected_failing_value)
            # The following options can be None:
            #   short_flag
            #   default_value
            #   expected_failing_value
            # For options with choices, list the test_values in a list of strings
            OptionStruct("string", "profile", "--config-file", "-f", "baz", None, None),
            OptionStruct("string", "profile", "--checkpoint-directory", "-s", "./test_dir", os.path.join(os.getcwd(), "checkpoints"), None),
            OptionStruct("string", "profile", "--output-model-repository-path", None, "./test_dir", os.path.join(os.getcwd(), "output_model_repository"), None),
            OptionStruct("string", "profile", "--client-protocol", None, ["http", "grpc"], "grpc", "SHOULD_FAIL"),
            OptionStruct("string", "profile", "--perf-analyzer-path", None, ".", "perf_analyzer", None),
            OptionStruct("string", "profile", "--perf-output-path", None, ".", None, None),
            OptionStruct("string", "profile", "--triton-docker-image", None, "test_image", DEFAULT_TRITON_DOCKER_IMAGE, None),
            OptionStruct("string", "profile", "--triton-http-endpoint", None, "localhost:4000", "localhost:8000", None),
            OptionStruct("string", "profile", "--triton-grpc-endpoint", None, "localhost:4001", "localhost:8001", None),
            OptionStruct("string", "profile", "--triton-metrics-url", None, "localhost:4002", "http://localhost:8002/metrics", None),
            OptionStruct("string", "profile", "--triton-server-path", None, "test_path", "tritonserver", None),
            OptionStruct("string", "profile", "--triton-output-path", None, "test_path", None, None),
            OptionStruct("string", "profile", "--triton-launch-mode", None, ["local", "docker", "remote","c_api"], "local", "SHOULD_FAIL"),

            #List of Strings Options:
            # Options format:
            #   (intlist/stringlist, MA step, long_flag, short_flag, test_value, default_value)
            # The following options can be None:
            #   short_flag
            #   default_value
            OptionStruct("intlist", "profile", "--batch-sizes", "-b", "2, 4, 6", "1"),
            OptionStruct("intlist", "profile", "--concurrency", "-c", "1, 2, 3", None),
            OptionStruct("stringlist", "profile", "--triton-docker-mounts", None, "a:b:c, d:e:f", None, extra_commands=["--triton-launch-mode", "docker"]),
            OptionStruct("stringlist", "profile", "--gpus", None, "a, b, c", "all"),
        ]
        #yapf: enable

        for option in options:
            self._resolve_test_values(option)

    def _resolve_test_values(self, option):
        if option.type in ["bool"]:
            self._test_boolean_option(option)
        elif option.type in ["int", "float"]:
            self._test_numeric_option(option)
        elif option.type in ["string"]:
            self._test_string_option(option)
        elif option.type in ["intlist", "stringlist"]:
            self._test_list_option(option)

    def _test_boolean_option(self, option_struct):
        option = option_struct.long_flag
        option_with_underscores = self._convert_flag(option)
        # print(f"\n>>> {option}")
        cli = CLIConfigStruct()
        cli = option_struct.cli_subcommand()
        _, config = cli.parse()
        option_value = config.get_config().get(option_with_underscores).value()
        self.assertEqual(option_value, False)

        # Test boolean option
        cli = CLIConfigStruct()
        cli.args.extend([option])
        _, config = cli.parse()
        option_value = config.get_config().get(option_with_underscores).value()
        self.assertEqual(option_value, True)

        # Test boolean option followed by value fails
        cli = CLIConfigStruct()
        cli.args.extend([option, 'SHOULD_FAIL'])
        with self.assertRaises(SystemExit):
            _, config = cli.parse()

    def _test_numeric_option(self, option_struct):
        long_option = option_struct.long_flag
        short_option = option_struct.short_flag
        expected_value_string = option_struct.expected_value
        expected_value = self._convert_string_to_numeric(
            option_struct.expected_value)
        default_value = self._convert_string_to_numeric(
            option_struct.default_value)

        # print(
        #     f"\n>>> {long_option},{short_option}, {expected_value}, {default_value}"
        # )
        long_option_with_underscores = self._convert_flag(long_option)

        # print(f"\t>>> long option flag: {long_option}, {expected_value}")
        # Test long_flag
        cli = CLIConfigStruct()
        cli.args.extend([long_option, expected_value_string])
        _, config = cli.parse()
        option_value = config.get_config().get(
            long_option_with_underscores).value()
        self.assertEqual(option_value, expected_value)

        # Test short_flag
        if short_option is not None:
            # print(f"\t>>> short option flag: {short_option}, {expected_value}")
            cli = CLIConfigStruct()
            cli.args.extend([short_option, expected_value_string])
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).value()
            self.assertEqual(option_value, expected_value)

        # Test default value for option
        if default_value is not None:
            # print(f"\t>>> default value: {long_option}, {default_value}")
            cli = CLIConfigStruct()
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).default_value()
            self.assertEqual(option_value, default_value)

    def _test_string_option(self, option_struct):
        long_option = option_struct.long_flag
        short_option = option_struct.short_flag
        expected_value = option_struct.expected_value
        default_value = option_struct.default_value
        expected_failing_value = option_struct.expected_failing_value

        # This covers strings that have choices
        # Recursively call this method with choices
        # print(f"expected value: {expected_value}, {type(expected_value)}")
        if type(expected_value) is list:
            for value in expected_value:
                new_struct = copy.deepcopy(option_struct)
                new_struct.expected_value = value
                self._test_string_option(new_struct)
        else:
            # print(
            #     f"\n>>> {long_option}, {short_option}, {expected_value}, {default_value}, {expected_failing_value}"
            # )
            long_option_with_underscores = self._convert_flag(long_option)

            # print(f"\t>>> long option flag: {long_option}, {expected_value}")
            # Test long flag
            cli = CLIConfigStruct()
            cli.args.extend([long_option, expected_value])
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).value()
            self.assertEqual(option_value, expected_value)

            # Test short flag
            if short_option is not None:
                # print(f"\t>>> short option flag: {short_option}, {expected_value}")
                cli = CLIConfigStruct()
                cli.args.extend([short_option, expected_value])
                _, config = cli.parse()
                option_value = config.get_config().get(
                    long_option_with_underscores).value()
                self.assertEqual(option_value, expected_value)

            # Test default value for option
            if default_value is not None:
                # print(f"\t>>> default value: {long_option}, {default_value}")
                cli = CLIConfigStruct()
                _, config = cli.parse()
                option_value = config.get_config().get(
                    long_option_with_underscores).default_value()
                self.assertEqual(option_value, default_value)

            # Verify that a incorrect value causes a failure
            if expected_failing_value is not None:
                # print(f"\t>>> error value: {long_option}, {expected_failing_value}")
                cli = CLIConfigStruct()
                cli.args.extend([long_option, expected_failing_value])
                with self.assertRaises(SystemExit):
                    _, config = cli.parse()

    def _test_list_option(self, option_struct):
        long_option = option_struct.long_flag
        short_option = option_struct.short_flag
        expected_value = option_struct.expected_value
        default_value = option_struct.default_value

        # Convert expected and default values to proper types for comparison
        if option_struct.type == "intlist":
            expected_value_converted = self._convert_string_to_int_list(
                expected_value)
            if default_value is not None:
                default_value_converted = self._convert_string_to_int_list(
                    default_value)
        else:
            expected_value_converted = expected_value.split(",")
            expected_value_converted = self._convert_string_to_string_list(
                expected_value)
            if default_value is not None:
                default_value_converted = self._convert_string_to_string_list(
                    default_value)

        long_option_with_underscores = self._convert_flag(long_option)

        # print(f"\t>>> long option flag: {long_option}, {expected_value}")
        # Test the long flag
        cli = CLIConfigStruct()
        cli.args.extend([long_option, expected_value])
        # print(f"extra commands: {option_struct.extra_commands}")
        if option_struct.extra_commands is not None:
            cli.args.extend(option_struct.extra_commands)
        _, config = cli.parse()
        option_value = config.get_config().get(
            long_option_with_underscores).value()
        self.assertEqual(option_value, expected_value_converted)

        # Test the short flag
        if short_option is not None:
            # print(f"\t>>> short option flag: {short_option}, {expected_value}")
            cli = CLIConfigStruct()
            cli.args.extend([short_option, expected_value])
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).value()
            self.assertEqual(option_value, expected_value_converted)

        # Verify the default value for the option
        if default_value is not None:
            # print(f"\t>>> default value: {long_option}, {default_value}")
            cli = CLIConfigStruct()
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).default_value()
            self.assertEqual(option_value, default_value_converted)

    # Helper methods
    def _convert_flag(self, option):
        return option.lstrip("-").replace("-", "_")

    def _convert_string_to_numeric(self, number):
        return float(number) if "." in number else int(number)

    def _convert_string_to_int_list(self, list_values):
        ret_val = [int(x) for x in list_values.split(",")]
        if len(ret_val) == 1:
            return ret_val[0]
        return ret_val

    def _convert_string_to_string_list(self, list_values):
        ret_val = [x for x in list_values.split(",")]
        if len(ret_val) == 1:
            return ret_val[0]
        return ret_val