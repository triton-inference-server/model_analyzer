#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        self._models = config["profile_models"]
        self._analyzer_log = analyzer_log

        check_function = self.__getattribute__(f"check_{test_name}")

        if check_function():
            sys.exit(0)
        else:
            sys.exit(1)

    def check_profile_logs(self):
        """
        Check that each model was profiled the number of times
        corresponding with batch size and concurrency combinations

        For multi-model with 2 models (A and B), the expected number of measurements
        is X*Y, where:
        - X is the number of measurements that we would expected
        for model A in a single-model case
        - Y is the number of measurements that we would expected
        for model B in a single-model case

        Since we know that X == Y for this test (since they both have the same
        parameters), then the expected total is just X*X
        """

        with open(self._analyzer_log, "r") as f:
            log_contents = f.read()

        expected_single_model_num_measurements = len(self._config["batch_sizes"]) * len(
            self._config["concurrency"]
        )
        expected_num_measurements = (
            expected_single_model_num_measurements
            * expected_single_model_num_measurements
        )
        for model in self._models:
            token = f"Profiling {model}_config_default"
            token_idx = 0
            found_count = 0
            while True:
                token_idx = log_contents.find(token, token_idx + 1)
                if token_idx == -1:
                    break
                found_count += 1
            if found_count != expected_num_measurements:
                print(
                    f"\n***\n***  Expected number of measurements for {model} : {expected_num_measurements}."
                    f"Found {found_count}. \n***"
                )
                return False
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--config-file",
        type=str,
        required=True,
        help="The path to the config yaml file.",
    )
    parser.add_argument(
        "-l",
        "--analyzer-log-file",
        type=str,
        required=True,
        help="The full path to the analyzer log.",
    )
    parser.add_argument(
        "-t",
        "--test-name",
        type=str,
        required=True,
        help="The name of the test to be run.",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    TestOutputValidator(config, args.test_name, args.analyzer_log_file)
