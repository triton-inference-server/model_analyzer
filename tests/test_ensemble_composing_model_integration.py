#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from model_analyzer.config.generate.model_profile_spec import ModelProfileSpec
from model_analyzer.config.generate.run_config_generator_factory import (
    RunConfigGeneratorFactory,
)
from model_analyzer.config.generate.search_dimension import SearchDimension
from model_analyzer.config.input.objects.config_model_profile_spec import (
    ConfigModelProfileSpec,
)


class TestEnsembleComposingModelIntegration(unittest.TestCase):
    """
    Integration tests for ensemble models with composing models that have
    instance_group count ranges in Quick search mode.
    """

    def setUp(self):
        """Create temporary model directories for testing"""
        self.temp_dir = tempfile.mkdtemp()

    def _create_mock_model_dir(self, model_name, supports_batching=True):
        """Helper to create a mock model directory"""
        model_path = os.path.join(self.temp_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        # Create a minimal config.pbtxt with batching support
        config_path = os.path.join(model_path, "config.pbtxt")
        with open(config_path, "w") as f:
            f.write(f'name: "{model_name}"\n')
            if supports_batching:
                f.write("max_batch_size: 128\n")
            else:
                f.write("max_batch_size: 0\n")
        return model_path

    def test_search_dimensions_created_for_cpu_and_gpu_composing_models(self):
        """
        Test that SearchDimensions are correctly created for an ensemble
        with CPU tokenizer and GPU inference model.
        """
        # Create mock model directories
        self._create_mock_model_dir("tokenizer")
        self._create_mock_model_dir("inference_model")

        # Create mock config with model repository
        config = MagicMock()
        config.model_repository = self.temp_dir
        config.triton_launch_mode = "local"

        # Create composing models with instance count ranges
        tokenizer_spec = ConfigModelProfileSpec(
            "tokenizer",
            model_config_parameters={
                "instance_group": [
                    [{"kind": "KIND_CPU", "count": [1, 2, 4, 8, 16, 32]}]
                ]
            },
        )

        inference_spec = ConfigModelProfileSpec(
            "inference_model",
            model_config_parameters={
                "instance_group": [[{"kind": "KIND_GPU", "count": [1, 2, 4, 8]}]]
            },
        )

        # Create ModelProfileSpec instances
        # Note: cpu_only needs to be set in config.cpu_only_composing_models
        config.cpu_only_composing_models = ["tokenizer"]

        tokenizer_model = ModelProfileSpec(tokenizer_spec, config, None, [])
        inference_model = ModelProfileSpec(inference_spec, config, None, [])

        # Create SearchConfig using the factory
        search_config = RunConfigGeneratorFactory._create_search_config(
            models=[], composing_models=[tokenizer_model, inference_model]
        )

        # Get dimensions
        dimensions = search_config.get_dimensions()

        # Should have 4 dimensions total:
        # - tokenizer: max_batch_size, instance_count
        # - inference_model: max_batch_size, instance_count
        self.assertEqual(len(dimensions), 4)

        # Check tokenizer dimensions (index 0 and 1)
        tokenizer_batch_dim = dimensions[0]
        tokenizer_instance_dim = dimensions[1]

        self.assertEqual(tokenizer_batch_dim.get_name(), "max_batch_size")
        self.assertEqual(
            tokenizer_batch_dim._type, SearchDimension.DIMENSION_TYPE_EXPONENTIAL
        )

        self.assertEqual(tokenizer_instance_dim.get_name(), "instance_count")
        self.assertEqual(
            tokenizer_instance_dim._type, SearchDimension.DIMENSION_TYPE_EXPONENTIAL
        )
        self.assertEqual(tokenizer_instance_dim.get_min_idx(), 0)  # 2^0 = 1
        self.assertEqual(tokenizer_instance_dim.get_max_idx(), 5)  # 2^5 = 32

        # Check inference model dimensions (index 2 and 3)
        inference_batch_dim = dimensions[2]
        inference_instance_dim = dimensions[3]

        self.assertEqual(inference_batch_dim.get_name(), "max_batch_size")
        self.assertEqual(inference_instance_dim.get_name(), "instance_count")
        self.assertEqual(
            inference_instance_dim._type, SearchDimension.DIMENSION_TYPE_EXPONENTIAL
        )
        self.assertEqual(inference_instance_dim.get_min_idx(), 0)  # 2^0 = 1
        self.assertEqual(inference_instance_dim.get_max_idx(), 3)  # 2^3 = 8

    def test_search_dimensions_with_linear_sequence(self):
        """
        Test that SearchDimensions handle contiguous linear sequences.
        """
        # Create mock model directory (no batching support)
        self._create_mock_model_dir("linear_model", supports_batching=False)

        # Create mock config with model repository
        config = MagicMock()
        config.model_repository = self.temp_dir
        config.triton_launch_mode = "local"
        config.cpu_only_composing_models = ["linear_model"]

        # Create model with linear instance count sequence
        model_spec = ConfigModelProfileSpec(
            "linear_model",
            model_config_parameters={
                "instance_group": [[{"kind": "KIND_CPU", "count": [1, 2, 3, 4, 5]}]]
            },
        )

        model = ModelProfileSpec(model_spec, config, None, [])

        search_config = RunConfigGeneratorFactory._create_search_config(
            models=[], composing_models=[model]
        )

        dimensions = search_config.get_dimensions()

        # Should have 1 dimension (no batching support)
        self.assertEqual(len(dimensions), 1)

        instance_dim = dimensions[0]
        self.assertEqual(instance_dim.get_name(), "instance_count")
        self.assertEqual(instance_dim._type, SearchDimension.DIMENSION_TYPE_LINEAR)
        self.assertEqual(instance_dim.get_min_idx(), 0)  # LINEAR: idx+1, so 0+1 = 1
        self.assertEqual(instance_dim.get_max_idx(), 4)  # LINEAR: idx+1, so 4+1 = 5

        # Verify it produces correct values
        self.assertEqual(instance_dim.get_value_at_idx(0), 1)
        self.assertEqual(instance_dim.get_value_at_idx(4), 5)

    def test_coordinate_values_map_to_correct_instance_counts(self):
        """
        Test that coordinate values correctly map to the intended instance counts
        from user-specified lists.
        """
        # Create mock model directory (no batching support)
        self._create_mock_model_dir("test_model", supports_batching=False)

        # Create mock config with model repository
        config = MagicMock()
        config.model_repository = self.temp_dir
        config.triton_launch_mode = "local"
        config.cpu_only_composing_models = []

        # Model with specific instance counts [2, 4, 8, 16]
        model_spec = ConfigModelProfileSpec(
            "test_model",
            model_config_parameters={
                "instance_group": [[{"kind": "KIND_GPU", "count": [2, 4, 8, 16]}]]
            },
        )

        model = ModelProfileSpec(model_spec, config, None, [])

        search_config = RunConfigGeneratorFactory._create_search_config(
            models=[], composing_models=[model]
        )

        dimensions = search_config.get_dimensions()
        instance_dim = dimensions[0]

        # The dimension should be EXPONENTIAL with min=1, max=4
        # because [2, 4, 8, 16] = [2^1, 2^2, 2^3, 2^4]
        self.assertEqual(instance_dim._type, SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        self.assertEqual(instance_dim.get_min_idx(), 1)
        self.assertEqual(instance_dim.get_max_idx(), 4)

        # Verify the values
        self.assertEqual(instance_dim.get_value_at_idx(1), 2)
        self.assertEqual(instance_dim.get_value_at_idx(2), 4)
        self.assertEqual(instance_dim.get_value_at_idx(3), 8)
        self.assertEqual(instance_dim.get_value_at_idx(4), 16)

    def test_ensemble_with_fixed_and_ranged_params(self):
        """
        Test ensemble where one composing model has instance range
        and the other has a fixed instance count.
        """
        # Create mock model directories (no batching support)
        self._create_mock_model_dir("ranged_model", supports_batching=False)
        self._create_mock_model_dir("fixed_model", supports_batching=False)

        # Create mock config with model repository
        config = MagicMock()
        config.model_repository = self.temp_dir
        config.triton_launch_mode = "local"
        config.cpu_only_composing_models = ["ranged_model"]

        # Model with range
        ranged_spec = ConfigModelProfileSpec(
            "ranged_model",
            model_config_parameters={
                "instance_group": [[{"kind": "KIND_CPU", "count": [1, 2, 4]}]]
            },
        )

        # Model without range (will use default)
        fixed_spec = ConfigModelProfileSpec("fixed_model")

        ranged_model = ModelProfileSpec(ranged_spec, config, None, [])
        fixed_model = ModelProfileSpec(fixed_spec, config, None, [])

        search_config = RunConfigGeneratorFactory._create_search_config(
            models=[], composing_models=[ranged_model, fixed_model]
        )

        dimensions = search_config.get_dimensions()

        # Should have 2 dimensions (one for each model)
        self.assertEqual(len(dimensions), 2)

        # First model has constrained dimension
        ranged_dim = dimensions[0]
        self.assertEqual(ranged_dim.get_min_idx(), 0)
        self.assertEqual(ranged_dim.get_max_idx(), 2)

        # Second model has unbounded dimension (default)
        fixed_dim = dimensions[1]
        self.assertEqual(fixed_dim.get_min_idx(), 0)
        self.assertEqual(fixed_dim.get_max_idx(), SearchDimension.DIMENSION_NO_MAX)

    def test_warning_for_non_existent_model_in_ensemble_composing_models(self):
        """
        Test that specifying a non-existent model in ensemble_composing_models
        logs a warning but doesn't fail.
        """
        from unittest.mock import patch

        # Create mock ensemble with only "model_a" and "model_b"
        self._create_mock_model_dir("ensemble_model")
        self._create_mock_model_dir("model_a")
        self._create_mock_model_dir("model_b")

        # Create ensemble config with only model_a and model_b
        ensemble_config = os.path.join(self.temp_dir, "ensemble_model", "config.pbtxt")
        with open(ensemble_config, "w") as f:
            f.write('name: "ensemble_model"\n')
            f.write('platform: "ensemble"\n')
            f.write("ensemble_scheduling {\n")
            f.write('  step { model_name: "model_a" model_version: -1 }\n')
            f.write('  step { model_name: "model_b" model_version: -1 }\n')
            f.write("}\n")

        # Mock config with ensemble_composing_models including non-existent "model_c"
        mock_config = MagicMock()
        mock_config.model_repository = self.temp_dir
        mock_config.ensemble_composing_models = [
            ConfigModelProfileSpec(
                "model_a",
                {"instance_group": [[{"kind": "KIND_GPU", "count": [1, 2, 4]}]]},
            ),
            ConfigModelProfileSpec("model_c"),  # This model doesn't exist in ensemble
        ]

        mock_client = MagicMock()
        mock_client.get_model_config.return_value = None

        mock_model = MagicMock()
        mock_model.model_name.return_value = "ensemble_model"
        mock_model.cpu_only.return_value = False

        # Capture log warnings
        with patch(
            "model_analyzer.config.generate.run_config_generator_factory.logger"
        ) as mock_logger:
            composing_models = (
                RunConfigGeneratorFactory._create_ensemble_composing_models(
                    mock_model, mock_config, mock_client, []
                )
            )

            # Should create models for model_a and model_b (auto-discovered)
            # model_c should be ignored with a warning
            self.assertEqual(len(composing_models), 2)
            composing_model_names = [m.model_name() for m in composing_models]
            self.assertIn("model_a", composing_model_names)
            self.assertIn("model_b", composing_model_names)
            self.assertNotIn("model_c", composing_model_names)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_message = mock_logger.warning.call_args[0][0]
            self.assertIn("model_c", warning_message)
            self.assertIn("not found", warning_message)
            self.assertIn("ensemble_model", warning_message)

    def test_explicit_kind_cpu_in_instance_group_sets_cpu_only(self):
        """
        Test that specifying KIND_CPU explicitly in instance_group sets
        _cpu_only = True without needing cpu_only_composing_models.
        """
        # Create mock model directory
        self._create_mock_model_dir("tokenizer")

        # Create mock config
        config = MagicMock()
        config.model_repository = self.temp_dir
        config.cpu_only_composing_models = []  # Empty - not using this option

        # Specify KIND_CPU explicitly in instance_group
        tokenizer_spec = ConfigModelProfileSpec(
            "tokenizer",
            model_config_parameters={
                "instance_group": [{"kind": "KIND_CPU", "count": [1, 2, 4]}]
            },
        )

        # Create ModelProfileSpec - should detect KIND_CPU and set cpu_only
        tokenizer_model = ModelProfileSpec(tokenizer_spec, config, None, [])

        # Verify cpu_only is True due to explicit KIND_CPU
        self.assertTrue(tokenizer_model.cpu_only())

    def test_explicit_kind_gpu_in_instance_group_sets_cpu_only_false(self):
        """
        Test that specifying KIND_GPU explicitly in instance_group sets
        _cpu_only = False even if model is in cpu_only_composing_models.
        """
        # Create mock model directory
        self._create_mock_model_dir("inference_model")

        # Create mock config - model IS in cpu_only_composing_models
        config = MagicMock()
        config.model_repository = self.temp_dir
        config.cpu_only_composing_models = [
            "inference_model"
        ]  # Would normally make it CPU-only

        # Specify KIND_GPU explicitly in instance_group (should override)
        inference_spec = ConfigModelProfileSpec(
            "inference_model",
            model_config_parameters={
                "instance_group": [{"kind": "KIND_GPU", "count": [1, 2, 4]}]
            },
        )

        # Create ModelProfileSpec - explicit KIND_GPU should override cpu_only_composing_models
        inference_model = ModelProfileSpec(inference_spec, config, None, [])

        # Verify cpu_only is False due to explicit KIND_GPU overriding cpu_only_composing_models
        self.assertFalse(inference_model.cpu_only())

    def test_no_explicit_kind_falls_back_to_cpu_only_composing_models(self):
        """
        Test that when no kind is specified, the cpu_only_composing_models
        config is still respected (backwards compatibility).
        """
        # Create mock model directory
        self._create_mock_model_dir("preprocessing")

        # Create mock config - model IS in cpu_only_composing_models
        config = MagicMock()
        config.model_repository = self.temp_dir
        config.cpu_only_composing_models = ["preprocessing"]

        # Don't specify kind in instance_group
        preprocessing_spec = ConfigModelProfileSpec(
            "preprocessing",
            model_config_parameters={
                "instance_group": [{"count": [1, 2, 4]}]  # No kind specified
            },
        )

        # Create ModelProfileSpec - should fall back to cpu_only_composing_models
        preprocessing_model = ModelProfileSpec(preprocessing_spec, config, None, [])

        # Verify cpu_only is True due to cpu_only_composing_models
        self.assertTrue(preprocessing_model.cpu_only())

    def test_no_explicit_kind_no_cpu_only_composing_models_defaults_to_gpu(self):
        """
        Test that when no kind is specified and model is not in
        cpu_only_composing_models, it defaults to GPU (cpu_only = False).
        """
        # Create mock model directory
        self._create_mock_model_dir("generic_model")

        # Create mock config - model NOT in cpu_only_composing_models
        config = MagicMock()
        config.model_repository = self.temp_dir
        config.cpu_only_composing_models = []  # Empty

        # Don't specify kind in instance_group
        generic_spec = ConfigModelProfileSpec(
            "generic_model",
            model_config_parameters={
                "instance_group": [{"count": [1, 2, 4]}]  # No kind specified
            },
        )

        # Create ModelProfileSpec - should default to GPU (cpu_only = False)
        generic_model = ModelProfileSpec(generic_spec, config, None, [])

        # Verify cpu_only is False (default)
        self.assertFalse(generic_model.cpu_only())

    def test_explicit_kind_priority_over_all_other_settings(self):
        """
        Test the priority order:
        1. Explicit kind in instance_group (highest)
        2. cpu_only_composing_models config
        3. Default to GPU (lowest)
        """
        # Test case: Model in cpu_only_composing_models but with explicit KIND_GPU
        self._create_mock_model_dir("mixed_model")

        config = MagicMock()
        config.model_repository = self.temp_dir
        config.cpu_only_composing_models = ["mixed_model"]

        # Explicit KIND_GPU should win over cpu_only_composing_models
        spec_with_gpu = ConfigModelProfileSpec(
            "mixed_model",
            model_config_parameters={
                "instance_group": [{"kind": "KIND_GPU", "count": [1, 2]}]
            },
        )

        model = ModelProfileSpec(spec_with_gpu, config, None, [])
        self.assertFalse(model.cpu_only())  # GPU wins

        # Same test with KIND_CPU
        spec_with_cpu = ConfigModelProfileSpec(
            "mixed_model",
            model_config_parameters={
                "instance_group": [{"kind": "KIND_CPU", "count": [1, 2]}]
            },
        )

        # Remove from cpu_only_composing_models to verify explicit KIND_CPU still works
        config.cpu_only_composing_models = []
        model_cpu = ModelProfileSpec(spec_with_cpu, config, None, [])
        self.assertTrue(
            model_cpu.cpu_only()
        )  # Explicit CPU works without cpu_only_composing_models

    def test_explicit_kind_with_double_wrapped_instance_group(self):
        """
        Test that explicit kind detection works with the double-wrapped
        instance_group structure produced by the config parser.

        The config parser produces: [[{'kind': 'KIND_CPU', 'count': [1, 2, 4]}]]
        (note the outer list wrapping from config parsing).
        """
        self._create_mock_model_dir("runtime_model")

        config = MagicMock()
        config.model_repository = self.temp_dir
        config.cpu_only_composing_models = []

        # Use the double-wrapped structure that the config parser produces at runtime
        spec_double_wrapped = ConfigModelProfileSpec(
            "runtime_model",
            model_config_parameters={
                # Note: Double wrapped [[{...}]] like config parser produces
                "instance_group": [[{"kind": "KIND_CPU", "count": [1, 2, 4]}]]
            },
        )

        model = ModelProfileSpec(spec_double_wrapped, config, None, [])

        # Verify cpu_only is True due to explicit KIND_CPU in double-wrapped structure
        self.assertTrue(model.cpu_only())

    def test_explicit_kind_with_list_wrapped_kind_value(self):
        """
        Test that explicit kind detection works when the kind value itself
        is wrapped in a list by the config parser.

        The full config parser structure is:
        [[{'kind': ['KIND_CPU'], 'count': [1, 2, 4]}]]
        (note: kind is ['KIND_CPU'] not 'KIND_CPU')
        """
        self._create_mock_model_dir("list_kind_model")

        config = MagicMock()
        config.model_repository = self.temp_dir
        config.cpu_only_composing_models = []

        # Use the structure with list-wrapped kind value
        spec_list_wrapped_kind = ConfigModelProfileSpec(
            "list_kind_model",
            model_config_parameters={
                # Both double-wrapped instance_group AND list-wrapped kind
                "instance_group": [[{"kind": ["KIND_CPU"], "count": [1, 2, 4]}]]
            },
        )

        model = ModelProfileSpec(spec_list_wrapped_kind, config, None, [])

        # Verify cpu_only is True due to explicit KIND_CPU in list-wrapped form
        self.assertTrue(model.cpu_only())

        # Also test KIND_GPU with list-wrapped kind
        spec_gpu_list = ConfigModelProfileSpec(
            "list_kind_model",
            model_config_parameters={
                "instance_group": [[{"kind": ["KIND_GPU"], "count": [1, 2]}]]
            },
        )

        # Model in cpu_only_composing_models but with explicit KIND_GPU in list form
        config.cpu_only_composing_models = ["list_kind_model"]
        model_gpu = ModelProfileSpec(spec_gpu_list, config, None, [])

        # Verify cpu_only is False due to explicit KIND_GPU overriding cpu_only_composing_models
        self.assertFalse(model_gpu.cpu_only())


if __name__ == "__main__":
    unittest.main()
