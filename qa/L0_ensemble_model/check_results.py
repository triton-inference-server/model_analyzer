#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

        (No model config parameter combos expected here!)
        """

        with open(self._analyzer_log, "r") as f:
            log_contents = f.read()

        # Quick search (hill-climbing algorithm) with default search space
        # Model - ensemble:
        #   concurrency: 1 to 1024 (11) [default max_concurrency]
        #   max_batch_size: model-dependent (~8)
        #   instance_group: 1 to 5 (5) [default max_instance_count]
        # Composing models also have instance_group configurations
        #
        # Quick search explores the space using hill-climbing, starting from
        # a default configuration and moving to better neighbors until convergence.
        # Ensemble models have additional composing models that are profiled together.
        #
        # With default max values, the search space is large, resulting in
        # more measurements as the algorithm explores different configurations.
        #
        # Minimum number of measurements: 20
        # Maximum number of measurements: 80
        expected_min_num_measurements = 20
        expected_max_num_measurements = 80

        for model in self._models:
            token = f"Profiling {model}_config"
            token_idx = 0
            found_count = 0
            while True:
                token_idx = log_contents.find(token, token_idx + 1)
                if token_idx == -1:
                    break
                found_count += 1
            if (
                found_count < expected_min_num_measurements
                or found_count > expected_max_num_measurements
            ):
                print(
                    f"\n***\n***  Expected range of measurements for {model} : {expected_min_num_measurements} to {expected_max_num_measurements}. "
                    f"Found {found_count}. \n***"
                )
                return False
        return True

    def check_composing_model_ranges(self):
        """
        Check that the ensemble composing model parameter ranges test ran successfully.

        With constrained instance_group counts:
        - add: [1, 2, 4] (3 values) with KIND_CPU
        - sub: [1, 2] (2 values) with KIND_GPU

        The search space is smaller than the default, so we expect fewer
        measurements. Quick search should explore the constrained space
        and find configurations within the specified ranges.

        We verify:
        1. The profiling completed (configs were generated)
        2. The number of measurements is within expected range for constrained search
        3. The 'add' model uses KIND_CPU (explicit kind in instance_group)
        4. The 'sub' model uses KIND_GPU (explicit kind in instance_group)
        """
        with open(self._analyzer_log, "r") as f:
            log_contents = f.read()

        # With constrained composing model ranges, the search space is smaller
        # Expected measurements: 10-40 (fewer than unconstrained)
        expected_min_num_measurements = 10
        expected_max_num_measurements = 40

        # Check for ensemble model profiling
        token = "Profiling ensemble_add_sub_config"
        token_idx = 0
        found_count = 0
        while True:
            token_idx = log_contents.find(token, token_idx + 1)
            if token_idx == -1:
                break
            found_count += 1

        if found_count < expected_min_num_measurements:
            print(
                f"\n***\n***  Expected at least {expected_min_num_measurements} measurements "
                f"for ensemble_add_sub with composing model ranges. Found {found_count}. \n***"
            )
            return False

        if found_count > expected_max_num_measurements:
            print(
                f"\n***\n***  Expected at most {expected_max_num_measurements} measurements "
                f"for ensemble_add_sub with composing model ranges. Found {found_count}. \n***"
            )
            return False

        print(
            f"Composing model ranges test: Found {found_count} measurements "
            f"(expected {expected_min_num_measurements}-{expected_max_num_measurements})"
        )

        # Verify explicit instance_group kind is respected
        # The 'add' model should use KIND_CPU (specified in config)
        # The 'sub' model should use KIND_GPU (specified in config)
        if not self._check_model_instance_kind(log_contents, "add", "KIND_CPU"):
            return False
        if not self._check_model_instance_kind(log_contents, "sub", "KIND_GPU"):
            return False

        return True

    def _check_model_instance_kind(self, log_contents, model_name, expected_kind):
        """
        Check that a model's instance_group uses the expected kind.

        The log format is multi-line:
        [Model Analyzer] Creating model config: add_config_0
        [Model Analyzer]   Setting instance_group to [{'count': 1, 'kind': 'KIND_CPU'}]

        We need to find "Creating model config: {model_name}_config" lines
        and then look at the following instance_group setting.
        """
        import re

        # Find all model config creations and their instance_group settings
        # Pattern matches: "Creating model config: {model_name}_config" followed by
        # "Setting instance_group to [...'kind': 'KIND_XXX'...]" within a few lines
        #
        # Use negative lookahead to stop at the next "Creating model config" or
        # "Creating ensemble model config" line, so we don't match across different models.
        # The lookahead uses .* to account for log prefixes like "[Model Analyzer]"
        pattern = (
            rf"Creating model config: {model_name}_config[^\n]*\n"
            rf"(?:(?!.*Creating (?:ensemble )?model config:)[^\n]*\n)*?"  # Stop at next model config
            rf"[^\n]*Setting instance_group to \[.*?'kind': '([A-Z_]+)'"
        )
        matches = re.findall(pattern, log_contents, re.MULTILINE)

        if not matches:
            print(
                f"\n***\n***  Could not find instance_group settings for model '{model_name}' in log.\n***"
            )
            return False

        # Check that all occurrences use the expected kind
        wrong_kind_count = 0
        for kind in matches:
            if kind != expected_kind:
                wrong_kind_count += 1

        if wrong_kind_count > 0:
            print(
                f"\n***\n***  Model '{model_name}' expected {expected_kind} but found "
                f"{wrong_kind_count} occurrences with wrong kind. Kinds found: {set(matches)}\n***"
            )
            return False

        print(
            f"Model '{model_name}' correctly uses {expected_kind} "
            f"({len(matches)} occurrences verified)"
        )
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
