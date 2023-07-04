#!/usr/bin/env python3

# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class TestConfigGenerator:
    """
    This class contains functions that
    create configs for various test scenarios.

    TO ADD A TEST: Simply add a member function whose name starts
                    with 'generate'.
    """

    def __init__(self):
        test_functions = [
            self.__getattribute__(name)
            for name in dir(self)
            if name.startswith("generate")
        ]

        for test_function in test_functions:
            self.setUp()
            test_function()

    def setUp(self):
        pass

    def generate_config_summaries(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m",
            "--profile-models",
            type=str,
            required=True,
            help="The models to be profiled",
        )

        args = parser.parse_args()
        profile_models = args.profile_models.split(",")

        self.config = {"constraints": {"perf_latency_p99": {"max": 50}}}
        self.config["profile_models"] = profile_models

        self.config["num_top_model_configs"] = 2
        with open("config-summaries.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_config_detailed_reports(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m",
            "--report-model-configs",
            type=str,
            required=True,
            help="The model config files for this test",
        )
        args = parser.parse_args()
        self.config = {}
        self.config["report_model_configs"] = [
            f"{model}_config_0" for model in args.report_model_configs.split(",")
        ]
        self.config["output_formats"] = ["pdf"]
        with open("config-detailed-reports.yml", "w+") as f:
            yaml.dump(self.config, f)


if __name__ == "__main__":
    TestConfigGenerator()
