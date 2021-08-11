# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import yaml


class TestOutputValidator:
    """
    Functions that validate the output
    of the test
    """
    def __init__(self, config, test_name, analyzer_log):
        self._config = config
        self._models = config['profile_models']
        self._analyzer_log = analyzer_log

        check_function = self.__getattribute__(f'check_{test_name}')

        if check_function():
            sys.exit(0)
        else:
            sys.exit(1)

    def check_steps_stability(self):
        """
        Makes sure that there were the same number of
        configurations tried in each search iteration. 
        """

        with open(self._analyzer_log, 'r+') as f:
            log_contents = f.read()

        logs_for_iteration = log_contents.split(
            'Profiling server only metrics...')[1:]

        logs_for_model = logs_for_iteration[0].split(
            "config search for model:")[1:]
        expected_step_counts = []
        for model_log in logs_for_model:
            expected_step_counts.append(model_log.count('[Search Step]'))

        for i in range(1, 4):
            logs_for_model = logs_for_iteration[i].split(
                "config search for model:")[1:]
            for j, model_log in enumerate(logs_for_model):
                actual_step_count = model_log.count('[Search Step]')
                if abs(actual_step_count - expected_step_counts[j]) > 1:
                    print("\n***\n***  Expected number of search steps for "
                          f"{self._models[j]} : {expected_step_counts[j]}."
                          f"Took {actual_step_count}. \n***")
                    return False
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--config-file',
                        type=str,
                        required=True,
                        help='The path to the config yaml file.')
    parser.add_argument('-l',
                        '--analyzer-log-file',
                        type=str,
                        required=True,
                        help='The full path to the analyzer log.')
    parser.add_argument('-t',
                        '--test-name',
                        type=str,
                        required=True,
                        help='The name of the test to be run.')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    TestOutputValidator(config, args.test_name, args.analyzer_log_file)
