# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import yaml
import sys


class TestOutputValidator:
    """
    Functions that validate the output
    of the test
    """

    def __init__(self, config, profile_models, analyzer_log, triton_log):
        self._config = config
        self._profile_models = profile_models.split(',')
        self._analyzer_log = analyzer_log
        self._triton_log = triton_log

        check_functions = [
            self.__getattribute__(name)
            for name in dir(self)
            if name.startswith('check')
        ]

        passed_test = True
        for check_function in check_functions:
            passed_test &= check_function()

        if passed_test:
            sys.exit(0)
        else:
            sys.exit(1)

    def check_perf_global(self):
        """
        Checks if the perf output reflects the custom
        flags set.

        Returns
        -------
        True if test passes else False
        """

        if 'perf_analyzer_flags' in self._config and 'percentile' in self._config[
                'perf_analyzer_flags']:
            with open(self._analyzer_log, 'r') as f:
                contents = f.read()

            # In contents, search for "stabilizing with px latency"
            percentile = self._config['perf_analyzer_flags']['percentile']
            token = f"Stabilizing using p{percentile} latency"

            # Ensure the token appears the correct number of times in the output contents
            next_token_idx = 0
            for profile_model in self._profile_models:
                next_token_idx = contents.find(token, next_token_idx)
                if next_token_idx == -1:
                    print(
                        f"\n***\n***  Perf Analyzer not stabilizing on p{percentile} latency"
                        f"for {profile_model}. \n***")
                    return False
        return True

    def check_perf_mode_time_window(self):
        """
        Checks if the time_window mode can be applied properly

        Returns
        -------
        True if test passes else False
        """

        if 'perf_analyzer_flags' in self._config and 'measurement-mode' in self._config[
                'perf_analyzer_flags']:
            with open(self._analyzer_log, 'r') as f:
                contents = f.read()

            # In contents, search for "stabilizing with px latency"
            measurement_mode = self._config['perf_analyzer_flags'][
                'measurement-mode']
            assert (measurement_mode == 'time_windows')
            token = "time_windows"

            # Ensure the token appears in the text
            token_idx = contents.find(token)
            if token_idx == -1:
                return False
        return True

    def check_perf_per_model(self):
        """
        Checks if the perf output reflects the per
        model custom flags set.

        Returns
        -------
        True if test passes else False
        """

        for profile_model, config_model in self._config['profile_models'].items(
        ):
            if 'perf_analyzer_flags' in config_model:
                with open(self._analyzer_log, 'r') as f:
                    contents = f.read()

                # In contents, search for "stabilizing with px latency"
                percentile = config_model['perf_analyzer_flags']['percentile']
                token = f"Stabilizing using p{percentile} latency"
                if contents.find(token) == -1:
                    print(
                        f"\n***\n***  Perf Analyzer not stabilizing on p{percentile} latency"
                        f"for {profile_model}. \n***")
                    return False
        return True

    def check_triton_global(self):
        """
        Checks if the triton log reflects the custom
        flags set.

        Returns
        -------
        True if test passes else False
        """

        if 'triton_server_flags' in self._config:
            with open(self._triton_log, 'r') as f:
                contents = f.read()

            # Look for strict-model-config false
            next_token_idx = 0
            for profile_model in self._config['profile_models']:
                next_token_idx = contents.find('strict_model_config',
                                               next_token_idx)
                if next_token_idx == -1:
                    print(
                        f"\n***\n*** strict-model-config for model {profile_model} not found in Triton log.\n***"
                    )
                    return False
                line = contents[contents[:next_token_idx].rfind('\n'):contents.
                                find('\n', next_token_idx)]
                strict_model_config_val = bool(
                    int(line.replace(' ', '').split('|')[2]))
                if strict_model_config_val != self._config[
                        'triton_server_flags']['strict_model_config']:
                    print(
                        f"\n***\n*** strict-model-config value does not match for model {profile_model}.\n***"
                    )
                    return False
        return True

    def check_triton_per_model(self):
        """
        Checks if the triton log reflects the per
        model custom flags set.

        Returns
        -------
        True if test passes else False
        """

        if 'triton_server_flags' not in self._config \
                and 'perf_analyzer_flags' not in self._config:
            with open(self._triton_log, 'r') as f:
                contents = f.read()

            # Get all the exit_timeout values from config
            timeouts_from_config = []
            for config_model in self._config['profile_models']:
                if 'triton_server_flags' in config_model:
                    timeouts_from_config.append(
                        config_model['triton_server_flags'])
                else:
                    return True

            # Get all the exit_timeout tokens from the logs
            timeouts_from_log = []
            next_token_idx = 0
            for profile_model in self._profile_models:
                next_token_idx = contents.find('exit_timeout', next_token_idx)
                if next_token_idx == -1:
                    print(
                        f"\n***\n*** Timeout for model {profile_model} not found in Triton log.\n***"
                    )
                    return False
                line = contents[contents[:next_token_idx].rfind('\n'):contents.
                                find('\n', next_token_idx)]
                timeouts_from_log.append(
                    int(line.replace(' ', '').split('|')[2]))

            if timeouts_from_config == timeouts_from_log:
                return True
            else:
                print(
                    "\n***\n*** Timeouts in log do not match those in config."
                    f" Expected timeouts {timeouts_from_config}, found {timeouts_from_log}\n***"
                )
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--config-file',
                        type=str,
                        required=True,
                        help='The config file for this test')
    parser.add_argument('-m',
                        '--profile-models',
                        type=str,
                        required=True,
                        help='The models being used for this test')
    parser.add_argument('--analyzer-log-file',
                        type=str,
                        required=True,
                        help='The full path to the analyzer log')
    parser.add_argument('--triton-log-file',
                        type=str,
                        required=True,
                        help='The full path to the triton log')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    TestOutputValidator(config, args.profile_models, args.analyzer_log_file,
                        args.triton_log_file)
