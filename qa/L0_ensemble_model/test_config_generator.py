#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import yaml


class TestConfigGenerator:
    """
    This class contains functions that
    create configs for various test scenarios.

    The `setup` function does the work common to all tests

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
            self.setup()
            test_function()

    def setup(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m",
            "--profile-models",
            type=str,
            required=True,
            help="Comma separated list of models to be profiled",
        )

        args = parser.parse_args()
        self.config = {}
        self.config["profile_models"] = sorted(args.profile_models.split(","))

    def generate_config(self):
        with open("config.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_config_with_ensemble_composing_ranges(self):
        """
        Generate a config that specifies instance_group count ranges
        for ensemble composing models using the ensemble_composing_models option.

        This tests the Quick search mode's ability to optimize
        composing model configurations with user-specified parameter ranges.

        Also tests explicit instance_group kind support:
        - 'add' model uses KIND_CPU (tests explicit CPU kind detection)
        - 'sub' model uses KIND_GPU (tests explicit GPU kind)
        This verifies that explicit kind in instance_group is respected
        without needing the separate cpu_only_composing_models config.
        """
        # Only profile the ensemble - composing models will be auto-discovered
        # but we specify their configs via ensemble_composing_models
        self.config["profile_models"] = ["ensemble_add_sub"]

        # Specify configs for the auto-discovered composing models
        # Use dictionary format (model names as keys) like profile_models
        # Note: Using KIND_CPU for 'add' to test explicit kind detection
        self.config["ensemble_composing_models"] = {
            "add": {
                "model_config_parameters": {
                    "instance_group": [{"kind": "KIND_CPU", "count": [1, 2, 4]}],
                    "dynamic_batching": {"max_queue_delay_microseconds": [0]},
                }
            },
            "sub": {
                "model_config_parameters": {
                    "instance_group": [{"kind": "KIND_GPU", "count": [1, 2]}],
                    "dynamic_batching": {"max_queue_delay_microseconds": [0]},
                }
            },
        }

        # Force quick search mode
        self.config["run_config_search_mode"] = "quick"

        with open("config_composing_ranges.yml", "w+") as f:
            yaml.dump(self.config, f)


if __name__ == "__main__":
    TestConfigGenerator()
