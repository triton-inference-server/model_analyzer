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

from distutils.cmd import Command
from email.policy import default
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


class CLIConfigProfileStruct():
    """
    Struct class to hold the common variables shared between profile tests
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


class CLIConfigAnalyzeStruct():
    """
    Struct class to hold the common variables shared between analyze tests
    """

    def __init__(self):
        #yapf: disable
        self.args = [
            '/usr/local/bin/model-analyzer',
            'analyze',
            '--analysis-models',
            'a,b,c'
        ]
        #yapf: enable
        config_analyze = ConfigCommandAnalyze()
        self.cli = CLISubclass()
        self.cli.add_subcommand(cmd='analyze', help='', config=config_analyze)

    def parse(self):
        return self.cli.parse(self.args)


class CLIConfigReportStruct():
    """
    Struct class to hold the common variables shared between analyze tests
    """

    def __init__(self):
        #yapf: disable
        self.args = [
            '/usr/local/bin/model-analyzer',
            'report',
            '--report-model-configs',
            'a, b, c'
        ]
        #yapf: enable
        config_report = ConfigCommandReport()
        self.cli = CLISubclass()
        self.cli.add_subcommand(cmd='report', help='', config=config_report)

    def parse(self):
        return self.cli.parse(self.args)


class OptionStruct():

    def __init__(self,
                 type,
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
        self.type = type
        self.extra_commands = extra_commands

        if stage == "profile":
            self.cli_subcommand = CLIConfigProfileStruct
        elif stage == "analyze":
            self.cli_subcommand = CLIConfigAnalyzeStruct
        elif stage == "report":
            self.cli_subcommand = CLIConfigReportStruct


@patch('model_analyzer.config.input.config_command_profile.file_path_validator',
       lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS))
@patch(
    'model_analyzer.config.input.config_command_profile.binary_path_validator',
    lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS))
@patch('model_analyzer.config.input.config_command_analyze.file_path_validator',
       lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS))
@patch('model_analyzer.config.input.config_command_report.file_path_validator',
       lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS))
class TestCLIOptions(trc.TestResultCollector):
    """
    Tests the methods of the CLI class
    """

    def test_basic_cli_config_profile_options(self):
        """
        Test the minimal set of cli commands necessary to run Model Analyzer profile
        """
        cli = CLIConfigProfileStruct()
        _, config = cli.parse()
        model_repo = config.model_repository
        profile_model = config.profile_models[0].model_name()
        self.assertEqual('foo', model_repo)
        self.assertEqual('bar', profile_model)

    @patch(
        'model_analyzer.config.input.config_command_report.ConfigCommandReport._load_config_file'
    )
    @patch(
        'model_analyzer.config.input.config_command_report.ConfigCommandReport._preprocess_and_verify_arguments'
    )
    @patch(
        'model_analyzer.config.input.config_command_analyze.ConfigCommandAnalyze._load_config_file'
    )
    @patch(
        'model_analyzer.config.input.config_command_analyze.ConfigCommandAnalyze._preprocess_and_verify_arguments'
    )
    @patch(
        'model_analyzer.config.input.config_command_profile.ConfigCommandProfile._load_config_file'
    )
    def test_all_options(self, mocked_load_config_file_profile,
                         mocked_verify_args_analyze,
                         mocked_load_config_file_analyze,
                         mocked_verify_args_report,
                         mocked_load_config_file_report):

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
            OptionStruct("bool", "profile","--reload-model-disable"),

            #Int/Float options
            # Options format:
            #   (int/float, MA step, long_option, short_option, test_value, default_value)
            # The following options can be None:
            #   short_option
            #   default_value
            OptionStruct("int", "profile", "--client-max-retries", "-r", "125", "50"),
            OptionStruct("int", "profile", "--duration-seconds", "-d", "10", "3"),
            OptionStruct("int", "profile", "--perf-analyzer-timeout", None, "100", "600"),
            OptionStruct("int", "profile", "--perf-analyzer-max-auto-adjusts", None, "100", "10"),
            OptionStruct("int", "profile", "--run-config-search-max-concurrency", None, "100", "1024"),
            OptionStruct("int", "profile", "--run-config-search-max-instance-count", None, "10", "5"),
            OptionStruct("float", "profile", "--monitoring-interval", "-i", "10.0", "1.0"),
            OptionStruct("float", "profile", "--perf-analyzer-cpu-util", None, "10.0", str(psutil.cpu_count() * 80.0)),
            OptionStruct("int", "analyze", "--num-configs-per-model", None, "10", "3"),
            OptionStruct("int", "analyze", "--num-top-model-configs", None, "10", "0"),
            OptionStruct("int", "analyze", "--latency-budget", None, "200", None),
            OptionStruct("int", "analyze", "--min-throughput", None, "300", None),

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
            OptionStruct("string", "profile", "--triton-install-path", None, "test_path", "/opt/tritonserver", None),
            OptionStruct("string", "analyze", "--checkpoint-directory", "-s", "./test_dir", os.path.join(os.getcwd(), "checkpoints"), None),
            OptionStruct("string", "analyze", "--export-path", "-e", "./test_dir", os.getcwd(), None),
            OptionStruct("string", "analyze", "--filename-model-inference", None, "foo", "metrics-model-inference.csv", None),
            OptionStruct("string", "analyze", "--filename-model-gpu", None, "foo", "metrics-model-gpu.csv", None),
            OptionStruct("string", "analyze", "--filename-server-only", None, "foo", "metrics-server-only.csv", None),
            OptionStruct("string", "analyze", "--config-file", "-f", "baz", None, None),
            OptionStruct("string", "report", "--checkpoint-directory", "-s", "./test_dir", os.path.join(os.getcwd(), "checkpoints"), None),
            OptionStruct("string", "report", "--export-path", "-e", "./test_dir", os.getcwd(), None),
            OptionStruct("string", "report", "--config-file", "-f", "baz", None, None),

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
            OptionStruct("stringlist", "analyze", "--inference-output-fields", None, "a, b, c",
                "model_name,batch_size,concurrency,model_config_path,instance_group,satisfies_constraints,perf_throughput,perf_latency_p99"),
            OptionStruct("stringlist", "analyze", "--gpu-output-fields", None, "a, b, c",
                "model_name,gpu_uuid,batch_size,concurrency,model_config_path,instance_group,satisfies_constraints,gpu_used_memory,gpu_utilization,gpu_power_usage"),
            OptionStruct("stringlist", "analyze", "--server-output-fields", None, "a, b, c",
                "model_name,gpu_uuid,gpu_used_memory,gpu_utilization,gpu_power_usage"),

            # No OP Options:
            # Option format:
            # (noop, any MA step, long_flag,)
            # These commands arent tested directly but are here to ensure that
            # the count is correct for all options in the config.
            # Some of these are required to run the subcommand
            # Others are yaml only options
            OptionStruct("noop", "profile", "--model-repository"),
            OptionStruct("noop", "profile", "--profile-models"),
            OptionStruct("noop", "analyze", "--analysis-models"),
            OptionStruct("noop", "report", "--report-model-configs"),
            OptionStruct("noop", "report", "--output-formats", "-o", ["pdf", "csv", "png"], "pdf", "SHOULD_FAIL"),
            OptionStruct("noop", "yaml_profile", "constraints"),
            OptionStruct("noop", "yaml_profile", "objectives"),
            OptionStruct("noop", "yaml_profile", "triton_server_flags"),
            OptionStruct("noop", "yaml_profile", "perf_analyzer_flags"),
            OptionStruct("noop", "yaml_profile", "triton_docker_labels"),
            OptionStruct("noop", "yaml_profile", "triton_server_environment"),
            OptionStruct("noop", "yaml_analyze", "constraints"),
            OptionStruct("noop", "yaml_analyze", "objectives"),
            OptionStruct("noop", "yaml_analyze", "plots"),
        ]
        #yapf: enable

        all_tested_options_set = set()

        for option in options:
            all_tested_options_set.add(self._convert_flag(option.long_flag))

            if option.type in ["bool"]:
                self._test_boolean_option(option)
            elif option.type in ["int", "float"]:
                self._test_numeric_option(option)
            elif option.type in ["string"]:
                self._test_string_option(option)
            elif option.type in ["intlist", "stringlist"]:
                self._test_list_option(option)

        self._verify_all_options_tested(all_tested_options_set)

    def _verify_all_options_tested(self, all_tested_options_set):
        cli_option_set = set()

        # Get all of the options in the CLI Configs
        structs = [
            CLIConfigProfileStruct, CLIConfigAnalyzeStruct,
            CLIConfigReportStruct
        ]
        for struct in structs:
            cli = struct()
            _, config = cli.parse()
            for key in config.get_config().keys():
                cli_option_set.add(key)

        self.assertEqual(cli_option_set, all_tested_options_set)

    def _test_boolean_option(self, option_struct):
        option = option_struct.long_flag
        option_with_underscores = self._convert_flag(option)
        cli = option_struct.cli_subcommand()
        _, config = cli.parse()
        option_value = config.get_config().get(option_with_underscores).value()
        self.assertEqual(option_value, False)

        # Test boolean option
        cli = option_struct.cli_subcommand()
        cli.args.extend([option])
        _, config = cli.parse()
        option_value = config.get_config().get(option_with_underscores).value()
        self.assertEqual(option_value, True)

        # Test boolean option followed by value fails
        cli = option_struct.cli_subcommand()
        cli.args.extend([option, 'SHOULD_FAIL'])
        with self.assertRaises(SystemExit):
            _, config = cli.parse()

    def _test_numeric_option(self, option_struct):
        long_option = option_struct.long_flag
        short_option = option_struct.short_flag
        expected_value_string = option_struct.expected_value
        expected_value = self._convert_string_to_numeric(
            option_struct.expected_value)
        default_value = None if option_struct.default_value == None else self._convert_string_to_numeric(
            option_struct.default_value)

        long_option_with_underscores = self._convert_flag(long_option)

        # Test long_flag
        cli = option_struct.cli_subcommand()
        cli.args.extend([long_option, expected_value_string])
        _, config = cli.parse()
        option_value = config.get_config().get(
            long_option_with_underscores).value()
        self.assertEqual(option_value, expected_value)

        # Test short_flag
        if short_option is not None:
            cli = option_struct.cli_subcommand()
            cli.args.extend([short_option, expected_value_string])
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).value()
            self.assertEqual(option_value, expected_value)

        # Test default value for option
        if default_value is not None:
            cli = option_struct.cli_subcommand()
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
        if type(expected_value) is list:
            for value in expected_value:
                new_struct = copy.deepcopy(option_struct)
                new_struct.expected_value = value
                self._test_string_option(new_struct)
        else:
            long_option_with_underscores = self._convert_flag(long_option)

            # Test long flag
            cli = option_struct.cli_subcommand()
            cli.args.extend([long_option, expected_value])
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).value()
            self.assertEqual(option_value, expected_value)

            # Test short flag
            if short_option is not None:
                cli = option_struct.cli_subcommand()
                cli.args.extend([short_option, expected_value])
                _, config = cli.parse()
                option_value = config.get_config().get(
                    long_option_with_underscores).value()
                self.assertEqual(option_value, expected_value)

            # Test default value for option
            if default_value is not None:
                cli = option_struct.cli_subcommand()
                _, config = cli.parse()
                option_value = config.get_config().get(
                    long_option_with_underscores).default_value()
                self.assertEqual(option_value, default_value)

            # Verify that a incorrect value causes a failure
            if expected_failing_value is not None:
                cli = option_struct.cli_subcommand()
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

        # Test the long flag
        cli = option_struct.cli_subcommand()
        cli.args.extend([long_option, expected_value])
        if option_struct.extra_commands is not None:
            cli.args.extend(option_struct.extra_commands)
        _, config = cli.parse()
        option_value = config.get_config().get(
            long_option_with_underscores).value()
        self.assertEqual(option_value, expected_value_converted)

        # Test the short flag
        if short_option is not None:
            cli = option_struct.cli_subcommand()
            cli.args.extend([short_option, expected_value])
            _, config = cli.parse()
            option_value = config.get_config().get(
                long_option_with_underscores).value()
            self.assertEqual(option_value, expected_value_converted)

        # Verify the default value for the option
        if default_value is not None:
            cli = option_struct.cli_subcommand()
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
