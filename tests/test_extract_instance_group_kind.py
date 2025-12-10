#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from model_analyzer.config.generate.quick_run_config_generator import (
    QuickRunConfigGenerator,
)


class TestExtractInstanceGroupKind(unittest.TestCase):
    """
    Tests for the _extract_instance_group_kind helper method.
    Tests the method directly without needing a full generator instance.
    """

    def test_extract_kind_cpu(self):
        """Test extracting KIND_CPU from instance_group."""
        model_config_params = {
            "instance_group": [[{"kind": "KIND_CPU", "count": [1, 2, 4]}]]
        }

        kind = QuickRunConfigGenerator._extract_instance_group_kind(
            None, model_config_params
        )

        self.assertEqual(kind, "KIND_CPU")

    def test_extract_kind_gpu(self):
        """Test extracting KIND_GPU from instance_group."""
        model_config_params = {
            "instance_group": [[{"kind": "KIND_GPU", "count": [1, 2, 4, 8]}]]
        }

        kind = QuickRunConfigGenerator._extract_instance_group_kind(
            None, model_config_params
        )

        self.assertEqual(kind, "KIND_GPU")

    def test_extract_kind_no_instance_group(self):
        """Test that empty string is returned when no instance_group."""
        model_config_params = {
            "dynamic_batching": {"max_queue_delay_microseconds": [0]}
        }

        kind = QuickRunConfigGenerator._extract_instance_group_kind(
            None, model_config_params
        )

        self.assertEqual(kind, "")

    def test_extract_kind_empty_params(self):
        """Test that empty string is returned for empty params."""
        kind = QuickRunConfigGenerator._extract_instance_group_kind(None, {})

        self.assertEqual(kind, "")

    def test_extract_kind_none_params(self):
        """Test that empty string is returned for None params."""
        kind = QuickRunConfigGenerator._extract_instance_group_kind(None, None)

        self.assertEqual(kind, "")

    def test_extract_kind_from_flattened_structure(self):
        """Test extracting kind from single-level list structure."""
        model_config_params = {
            "instance_group": [{"kind": "KIND_CPU", "count": [2, 4, 8]}]
        }

        kind = QuickRunConfigGenerator._extract_instance_group_kind(
            None, model_config_params
        )

        # Should handle both nested [[{...}]] and flat [{...}] structures
        self.assertEqual(kind, "KIND_CPU")


if __name__ == "__main__":
    unittest.main()
