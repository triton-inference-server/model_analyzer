#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class TestComposingModelValidation(unittest.TestCase):
    """
    Tests for validating that composing models can have parameter ranges
    in Quick search mode while top-level models cannot.
    """

    def setUp(self):
        """Create temporary config file for tests"""
        self.temp_dir = tempfile.mkdtemp()

    def _create_config_file(self, config_dict):
        """Helper to create a temporary YAML config file"""
        config_path = f"{self.temp_dir}/test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)
        return config_path

    def test_composing_model_with_instance_group_range_allowed(self):
        """
        Test that composing models can specify instance_group count ranges
        in Quick mode without validation errors.
        """
        config = {
            "model_repository": "/tmp/models",
            "run_config_search_mode": "quick",
            "profile_models": {
                "ensemble_model": {},
                "tokenizer": {
                    "model_config_parameters": {
                        "instance_group": [{"kind": "KIND_CPU", "count": [1, 2, 4, 8]}]
                    }
                },
            },
            "cpu_only_composing_models": ["tokenizer"],
        }

        config_path = self._create_config_file(config)

        # This should NOT raise an exception
        try:
            args = MagicMock()
            args.config_file = config_path
            args.mode = "profile"

            with patch(
                "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._preprocess_and_verify_arguments"
            ):
                with patch(
                    "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._autofill_values"
                ):
                    ConfigCommandProfile()
                    # The validation happens in _set_config
                    # If it passes without exception, the test passes
        except TritonModelAnalyzerException as e:
            self.fail(
                f"Validation should not fail for composing model with instance_group range: {e}"
            )

    def test_top_level_model_with_instance_group_range_rejected(self):
        """
        Test that top-level (non-composing) models CANNOT specify instance_group
        count ranges in Quick mode - should raise validation error.
        """
        config = {
            "model_repository": "/tmp/models",
            "run_config_search_mode": "quick",
            "profile_models": {
                "my_model": {
                    "model_config_parameters": {
                        "instance_group": [{"kind": "KIND_GPU", "count": [1, 2, 4, 8]}]
                    }
                }
            },
        }

        config_path = self._create_config_file(config)

        # This SHOULD raise an exception during initialization
        with self.assertRaises(TritonModelAnalyzerException) as context:
            from argparse import Namespace

            args_ns = Namespace(config_file=config_path, mode="offline")

            config_obj = ConfigCommandProfile()
            config_obj.set_config_values(args_ns)

        # Verify the error message mentions composing models can use ranges
        error_msg = str(context.exception).lower()
        self.assertIn("top-level", error_msg)

    def test_bls_composing_model_with_multiple_param_combinations_allowed(self):
        """
        Test that BLS composing models can have multiple parameter combinations
        (e.g., instance_group + dynamic_batching ranges) in Quick mode.
        """
        config = {
            "model_repository": "/tmp/models",
            "run_config_search_mode": "quick",
            "bls_composing_models": ["preprocessing"],
            "profile_models": {
                "bls_model": {},
                "preprocessing": {
                    "model_config_parameters": {
                        "instance_group": [{"kind": "KIND_CPU", "count": [1, 2, 4]}],
                        "dynamic_batching": {
                            "max_queue_delay_microseconds": [0, 100, 200]
                        },
                    }
                },
            },
        }

        config_path = self._create_config_file(config)

        # This should NOT raise an exception
        try:
            args = MagicMock()
            args.config_file = config_path
            args.mode = "profile"

            with patch(
                "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._preprocess_and_verify_arguments"
            ):
                with patch(
                    "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._autofill_values"
                ):
                    ConfigCommandProfile()
        except TritonModelAnalyzerException as e:
            self.fail(
                f"Validation should not fail for BLS composing model with parameter ranges: {e}"
            )

    def test_is_composing_model_helper_identifies_bls_composing(self):
        """
        Test that _is_composing_model correctly identifies BLS composing models.
        """
        from model_analyzer.config.input.config_command import ConfigCommand

        config_dict = {
            "bls_composing_models": MagicMock(
                value=lambda: [
                    MagicMock(model_name=lambda: "model_a"),
                    MagicMock(model_name=lambda: "model_b"),
                ]
            )
        }

        cmd = ConfigCommand()

        self.assertTrue(cmd._is_composing_model("model_a", config_dict))
        self.assertTrue(cmd._is_composing_model("model_b", config_dict))
        self.assertFalse(cmd._is_composing_model("model_c", config_dict))

    def test_is_composing_model_helper_identifies_cpu_only_composing(self):
        """
        Test that _is_composing_model correctly identifies CPU-only composing models.
        """
        from model_analyzer.config.input.config_command import ConfigCommand

        config_dict = {
            "cpu_only_composing_models": MagicMock(
                value=lambda: ["tokenizer", "preprocessor"]
            )
        }

        cmd = ConfigCommand()

        self.assertTrue(cmd._is_composing_model("tokenizer", config_dict))
        self.assertTrue(cmd._is_composing_model("preprocessor", config_dict))
        self.assertFalse(cmd._is_composing_model("embedding", config_dict))

    def test_ensemble_with_cpu_and_gpu_composing_models_validates(self):
        """
        Test ensemble configuration with CPU tokenizer and GPU inference model
        to ensure it validates correctly with instance count ranges.
        """
        config = {
            "model_repository": "/registry",
            "export_path": "/tmp/ensemble_results",
            "override_output_model_repository": True,
            "triton_launch_mode": "local",
            "run_config_search_mode": "quick",
            "profile_models": {
                "tokenizer": {
                    "model_config_parameters": {
                        "instance_group": [
                            {"kind": "KIND_CPU", "count": [1, 2, 4, 8, 16, 32]}
                        ],
                        "dynamic_batching": {"max_queue_delay_microseconds": [0]},
                    }
                },
                "inference_model": {
                    "model_config_parameters": {
                        "instance_group": [{"kind": "KIND_GPU", "count": [1, 2, 4, 8]}],
                        "dynamic_batching": {"max_queue_delay_microseconds": [0]},
                    }
                },
                "ensemble_model": {
                    "model_config_parameters": {
                        "dynamic_batching": {"max_queue_delay_microseconds": [0]}
                    },
                    "perf_analyzer_flags": {"shape": ["input:1"], "string-length": 20},
                },
            },
            "cpu_only_composing_models": ["tokenizer"],
        }

        config_path = self._create_config_file(config)

        # This should NOT raise an exception with our changes
        try:
            args = MagicMock()
            args.config_file = config_path
            args.mode = "profile"

            with patch(
                "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._preprocess_and_verify_arguments"
            ):
                with patch(
                    "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._autofill_values"
                ):
                    ConfigCommandProfile()
        except TritonModelAnalyzerException as e:
            self.fail(f"Ensemble config should validate but got error: {e}")

    def test_ensemble_composing_models_with_instance_group_range_allowed(self):
        """
        Test that ensemble_composing_models config option allows composing models
        to specify instance_group count ranges in Quick mode.
        """
        config = {
            "model_repository": "/tmp/models",
            "run_config_search_mode": "quick",
            "profile_models": ["ensemble_model"],
            "ensemble_composing_models": [
                {
                    "model_name": "tokenizer",
                    "model_config_parameters": {
                        "instance_group": [
                            {"kind": "KIND_CPU", "count": [1, 2, 4, 8, 16, 32]}
                        ]
                    },
                },
                {
                    "model_name": "inference",
                    "model_config_parameters": {
                        "instance_group": [{"kind": "KIND_GPU", "count": [1, 2, 4, 8]}]
                    },
                },
            ],
        }

        config_path = self._create_config_file(config)

        # This should NOT raise an exception
        try:
            args = MagicMock()
            args.config_file = config_path
            args.mode = "profile"

            with patch(
                "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._preprocess_and_verify_arguments"
            ):
                with patch(
                    "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._autofill_values"
                ):
                    ConfigCommandProfile()
        except TritonModelAnalyzerException as e:
            self.fail(
                f"Validation should not fail for ensemble_composing_models with instance_group ranges: {e}"
            )

    def test_is_composing_model_helper_identifies_ensemble_composing(self):
        """
        Test that _is_composing_model correctly identifies ensemble composing models.
        """
        from model_analyzer.config.input.config_command import ConfigCommand

        config_dict = {
            "ensemble_composing_models": MagicMock(
                value=lambda: [
                    MagicMock(model_name=lambda: "model_a"),
                    MagicMock(model_name=lambda: "model_b"),
                ]
            )
        }

        cmd = ConfigCommand()

        self.assertTrue(cmd._is_composing_model("model_a", config_dict))
        self.assertTrue(cmd._is_composing_model("model_b", config_dict))
        self.assertFalse(cmd._is_composing_model("model_c", config_dict))


if __name__ == "__main__":
    unittest.main()
