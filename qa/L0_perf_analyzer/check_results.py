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

    def __init__(self, config, config_file, analyzer_log, test_name):
        self._config = config
        self._config_file = config_file
        self._analyzer_log = analyzer_log

        check_function = self.__getattribute__(f'check_{test_name}')

        if check_function():
            sys.exit(0)
        else:
            sys.exit(1)

    def check_time_window(self):
        """
        Look for no measurement window adjustment
        """
        with open(self._analyzer_log, "r") as f:
            if "perf_analyzer's measurement window is too small" in f.read():
                print(
                    "\n***\n*** Unexpected time window adjustment found.\n***")
                return False
        return True

    def check_time_window_adjust(self):
        """
        Look for measurement window adjustment
        """

        with open(self._analyzer_log, "r") as f:
            log_contents = f.read()

        if log_contents.find(
                "measurement window is too small, increased to") != -1:
            return True
        print("\n***\n*** Time window adjustment expected but not found.\n***")
        return False

    def check_count_window(self):
        """
        Look for no request count adjustment
        """

        with open(self._analyzer_log, "r") as f:
            if "perf_analyzer's measurement window is too small" in f.read():
                print(
                    "\n***\n*** Unexpected count window adjustment found.\n***")
                return False
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--config-file',
                        type=str,
                        required=True,
                        help='The path to the config yaml file.')
    parser.add_argument('--analyzer-log',
                        type=str,
                        required=True,
                        help='The path to the analyzer log file.')
    parser.add_argument('--test-name',
                        type=str,
                        required=True,
                        help='The name of the test to be run.')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    TestOutputValidator(config, args.config_file, args.analyzer_log,
                        args.test_name)
