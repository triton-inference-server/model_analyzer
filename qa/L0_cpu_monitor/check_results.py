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
import yaml
import sys


class TestOutputValidator:
    """
    Functions that validate the output
    of the test
    """

    def __init__(self, config, profile_models, analyzer_log, triton_log):
        self._config = config
        self._profile_models = profile_models.split(",")
        self._analyzer_log = analyzer_log
        self._triton_log = triton_log

        check_functions = [
            self.__getattribute__(name)
            for name in dir(self)
            if name.startswith("check")
        ]
        passed_test = True
        for check_function in check_functions:
            passed_test &= check_function()
        if passed_test:
            sys.exit(0)
        else:
            sys.exit(1)

    def check_cpu_monitor_run(self):
        """
        Check if CPU monitor had or had not run

        Returns
        -------
        True if test passes else False
        """
        run_statement = "CPU monitor is enabled for profiling."
        run_expectation = {
            "config-gpu-on-cpu-only-on-monitor-auto.yml.log": True,
            "config-gpu-on-cpu-only-on-monitor-off.yml.log": False,
            "config-gpu-on-cpu-only-on-monitor-on.yml.log": True,
            "config-gpu-on-cpu-only-off-monitor-auto.yml.log": False,
            "config-gpu-on-cpu-only-off-monitor-off.yml.log": False,
            "config-gpu-on-cpu-only-off-monitor-on.yml.log": True,
            "config-gpu-off-cpu-only-on-monitor-auto.yml.log": True,
            "config-gpu-off-cpu-only-on-monitor-off.yml.log": False,
            "config-gpu-off-cpu-only-on-monitor-on.yml.log": True,
            "config-gpu-off-cpu-only-off-monitor-auto.yml.log": True,
            "config-gpu-off-cpu-only-off-monitor-off.yml.log": False,
            "config-gpu-off-cpu-only-off-monitor-on.yml.log": True
        }
        with open(self._analyzer_log, "r") as f:
            return (run_statement
                    in f.read()) == run_expectation[self._analyzer_log]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--config-file",
                        type=str,
                        required=True,
                        help="The config file for this test")
    parser.add_argument("-m",
                        "--profile-models",
                        type=str,
                        required=True,
                        help="The models being used for this test")
    parser.add_argument("--analyzer-log-file",
                        type=str,
                        required=True,
                        help="The full path to the analyzer log")
    parser.add_argument("--triton-log-file",
                        type=str,
                        required=True,
                        help="The full path to the triton log")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    TestOutputValidator(config, args.profile_models, args.analyzer_log_file,
                        args.triton_log_file)
