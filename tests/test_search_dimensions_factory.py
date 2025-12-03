#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock

from model_analyzer.config.generate.run_config_generator_factory import (
    RunConfigGeneratorFactory,
)
from model_analyzer.config.generate.search_dimension import SearchDimension
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class TestSearchDimensionsFactory(unittest.TestCase):
    """
    Tests for SearchDimensions factory methods that handle user-specified
    instance_group count lists.
    """

    def test_get_instance_count_list_with_powers_of_two(self):
        """Test extracting instance count list from model config parameters."""
        model = MagicMock()
        model.model_config_parameters.return_value = {
            "instance_group": [[{"kind": "KIND_CPU", "count": [1, 2, 4, 8, 16, 32]}]]
        }

        result = RunConfigGeneratorFactory._get_instance_count_list(model)

        self.assertEqual(result, [1, 2, 4, 8, 16, 32])

    def test_get_instance_count_list_no_instance_group(self):
        """Test that empty list is returned when no instance_group specified."""
        model = MagicMock()
        model.model_config_parameters.return_value = {
            "dynamic_batching": {"max_queue_delay_microseconds": [0, 100]}
        }

        result = RunConfigGeneratorFactory._get_instance_count_list(model)

        self.assertEqual(result, [])

    def test_get_instance_count_list_no_model_config_params(self):
        """Test that empty list is returned when no model_config_parameters."""
        model = MagicMock()
        model.model_config_parameters.return_value = None

        result = RunConfigGeneratorFactory._get_instance_count_list(model)

        self.assertEqual(result, [])

    def test_get_instance_count_list_single_value_not_list(self):
        """Test that empty list is returned when count is a single value, not a list."""
        model = MagicMock()
        model.model_config_parameters.return_value = {
            "instance_group": [[{"kind": "KIND_GPU", "count": 4}]]
        }

        result = RunConfigGeneratorFactory._get_instance_count_list(model)

        self.assertEqual(result, [])

    def test_is_powers_of_two_valid_sequence(self):
        """Test detection of valid powers of 2 sequence."""
        self.assertTrue(
            RunConfigGeneratorFactory._is_powers_of_two([1, 2, 4, 8, 16, 32])
        )
        self.assertTrue(RunConfigGeneratorFactory._is_powers_of_two([2, 4, 8]))
        self.assertTrue(RunConfigGeneratorFactory._is_powers_of_two([1]))
        self.assertTrue(RunConfigGeneratorFactory._is_powers_of_two([1, 2]))

    def test_is_powers_of_two_invalid_sequence(self):
        """Test rejection of non-powers of 2."""
        self.assertFalse(RunConfigGeneratorFactory._is_powers_of_two([1, 2, 3, 4]))
        self.assertFalse(RunConfigGeneratorFactory._is_powers_of_two([1, 2, 4, 7, 16]))
        self.assertFalse(RunConfigGeneratorFactory._is_powers_of_two([0, 1, 2]))
        self.assertFalse(RunConfigGeneratorFactory._is_powers_of_two([-1, 1, 2]))

    def test_is_linear_sequence_valid(self):
        """Test detection of valid linear (contiguous) sequences."""
        self.assertTrue(RunConfigGeneratorFactory._is_linear_sequence([1, 2, 3, 4, 5]))
        self.assertTrue(RunConfigGeneratorFactory._is_linear_sequence([5, 6, 7]))
        self.assertTrue(RunConfigGeneratorFactory._is_linear_sequence([1]))
        self.assertTrue(RunConfigGeneratorFactory._is_linear_sequence([10, 11]))

    def test_is_linear_sequence_invalid(self):
        """Test rejection of non-contiguous sequences."""
        self.assertFalse(RunConfigGeneratorFactory._is_linear_sequence([1, 2, 4, 5]))
        self.assertFalse(RunConfigGeneratorFactory._is_linear_sequence([1, 3, 5, 7]))
        self.assertFalse(RunConfigGeneratorFactory._is_linear_sequence([1, 2, 3, 5]))

    def test_create_instance_dimension_from_powers_of_two(self):
        """Test creating EXPONENTIAL dimension from powers of 2 list."""
        count_list = [1, 2, 4, 8, 16, 32]

        dim = RunConfigGeneratorFactory._create_instance_dimension_from_list(count_list)

        self.assertEqual(dim.get_name(), "instance_count")
        self.assertEqual(dim._type, SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        self.assertEqual(dim.get_min_idx(), 0)  # 2^0 = 1
        self.assertEqual(dim.get_max_idx(), 5)  # 2^5 = 32

        # Verify that the dimension produces correct values
        self.assertEqual(dim.get_value_at_idx(0), 1)
        self.assertEqual(dim.get_value_at_idx(1), 2)
        self.assertEqual(dim.get_value_at_idx(2), 4)
        self.assertEqual(dim.get_value_at_idx(5), 32)

    def test_create_instance_dimension_from_subset_powers_of_two(self):
        """Test creating EXPONENTIAL dimension from subset of powers of 2."""
        count_list = [4, 8, 16]  # 2^2 to 2^4

        dim = RunConfigGeneratorFactory._create_instance_dimension_from_list(count_list)

        self.assertEqual(dim._type, SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        self.assertEqual(dim.get_min_idx(), 2)  # 2^2 = 4
        self.assertEqual(dim.get_max_idx(), 4)  # 2^4 = 16

    def test_create_instance_dimension_from_linear_sequence(self):
        """Test creating LINEAR dimension from contiguous sequence."""
        count_list = [1, 2, 3, 4, 5]

        dim = RunConfigGeneratorFactory._create_instance_dimension_from_list(count_list)

        self.assertEqual(dim.get_name(), "instance_count")
        self.assertEqual(dim._type, SearchDimension.DIMENSION_TYPE_LINEAR)
        self.assertEqual(dim.get_min_idx(), 0)  # LINEAR: idx+1, so 0+1 = 1
        self.assertEqual(dim.get_max_idx(), 4)  # LINEAR: idx+1, so 4+1 = 5

        # Verify that the dimension produces correct values
        self.assertEqual(dim.get_value_at_idx(0), 1)
        self.assertEqual(dim.get_value_at_idx(1), 2)
        self.assertEqual(dim.get_value_at_idx(4), 5)

    def test_create_instance_dimension_from_linear_sequence_not_starting_at_one(self):
        """Test creating LINEAR dimension from sequence not starting at 1."""
        count_list = [5, 6, 7, 8]

        dim = RunConfigGeneratorFactory._create_instance_dimension_from_list(count_list)

        self.assertEqual(dim._type, SearchDimension.DIMENSION_TYPE_LINEAR)
        self.assertEqual(dim.get_min_idx(), 4)  # LINEAR: idx+1, so 4+1 = 5
        self.assertEqual(dim.get_max_idx(), 7)  # LINEAR: idx+1, so 7+1 = 8

    def test_create_instance_dimension_from_invalid_list_raises_exception(self):
        """Test that arbitrary lists raise helpful exception."""
        count_list = [1, 3, 7, 15]  # Not powers of 2, not contiguous

        with self.assertRaises(TritonModelAnalyzerException) as context:
            RunConfigGeneratorFactory._create_instance_dimension_from_list(count_list)

        error_msg = str(context.exception)
        self.assertIn("not compatible with Quick search mode", error_msg)
        self.assertIn("powers of 2", error_msg)
        self.assertIn("contiguous sequence", error_msg)

    def test_create_instance_dimension_empty_list_raises_exception(self):
        """Test that empty list raises exception."""
        with self.assertRaises(TritonModelAnalyzerException) as context:
            RunConfigGeneratorFactory._create_instance_dimension_from_list([])

        self.assertIn("cannot be empty", str(context.exception))

    def test_get_dimensions_for_model_with_instance_count_list(self):
        """Test that _get_dimensions_for_model uses user-specified count list."""
        model = MagicMock()
        model.model_config_parameters.return_value = {
            "instance_group": [[{"kind": "KIND_CPU", "count": [1, 2, 4, 8]}]]
        }
        model.supports_batching.return_value = True

        dims = RunConfigGeneratorFactory._get_dimensions_for_model(model)

        # Should have 2 dimensions: max_batch_size and instance_count
        self.assertEqual(len(dims), 2)

        # First should be max_batch_size (default exponential)
        self.assertEqual(dims[0].get_name(), "max_batch_size")
        self.assertEqual(dims[0]._type, SearchDimension.DIMENSION_TYPE_EXPONENTIAL)

        # Second should be constrained instance_count from user list
        self.assertEqual(dims[1].get_name(), "instance_count")
        self.assertEqual(dims[1]._type, SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        self.assertEqual(dims[1].get_min_idx(), 0)  # 2^0 = 1
        self.assertEqual(dims[1].get_max_idx(), 3)  # 2^3 = 8

    def test_get_dimensions_for_model_without_instance_count_list(self):
        """Test that _get_dimensions_for_model uses default unbounded dimension."""
        model = MagicMock()
        model.model_config_parameters.return_value = {}
        model.supports_batching.return_value = True

        dims = RunConfigGeneratorFactory._get_dimensions_for_model(model)

        # Should have 2 dimensions: max_batch_size and instance_count
        self.assertEqual(len(dims), 2)

        # instance_count should be unbounded (default min=0, max=DIMENSION_NO_MAX)
        instance_dim = dims[1]
        self.assertEqual(instance_dim.get_name(), "instance_count")
        self.assertEqual(instance_dim._type, SearchDimension.DIMENSION_TYPE_LINEAR)
        self.assertEqual(instance_dim.get_min_idx(), 0)
        self.assertEqual(instance_dim.get_max_idx(), SearchDimension.DIMENSION_NO_MAX)

    def test_get_dimensions_for_model_no_batching_support(self):
        """Test that non-batching models get only instance_count dimension."""
        model = MagicMock()
        model.model_config_parameters.return_value = {
            "instance_group": [[{"kind": "KIND_CPU", "count": [2, 4, 8]}]]
        }
        model.supports_batching.return_value = False

        dims = RunConfigGeneratorFactory._get_dimensions_for_model(model)

        # Should have only 1 dimension: instance_count
        self.assertEqual(len(dims), 1)
        self.assertEqual(dims[0].get_name(), "instance_count")
        self.assertEqual(dims[0]._type, SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        self.assertEqual(dims[0].get_min_idx(), 1)  # 2^1 = 2
        self.assertEqual(dims[0].get_max_idx(), 3)  # 2^3 = 8

    def test_cpu_tokenizer_config_full_range(self):
        """Test CPU tokenizer with full powers-of-2 range [1, 2, 4, 8, 16, 32]."""
        model = MagicMock()
        model.model_config_parameters.return_value = {
            "instance_group": [[{"kind": "KIND_CPU", "count": [1, 2, 4, 8, 16, 32]}]]
        }
        model.supports_batching.return_value = True

        dims = RunConfigGeneratorFactory._get_dimensions_for_model(model)

        # Verify instance_count dimension
        instance_dim = [d for d in dims if d.get_name() == "instance_count"][0]
        self.assertEqual(instance_dim.get_min_idx(), 0)  # 2^0 = 1
        self.assertEqual(instance_dim.get_max_idx(), 5)  # 2^5 = 32

        # Verify it produces the correct values
        self.assertEqual(instance_dim.get_value_at_idx(0), 1)
        self.assertEqual(instance_dim.get_value_at_idx(1), 2)
        self.assertEqual(instance_dim.get_value_at_idx(2), 4)
        self.assertEqual(instance_dim.get_value_at_idx(3), 8)
        self.assertEqual(instance_dim.get_value_at_idx(4), 16)
        self.assertEqual(instance_dim.get_value_at_idx(5), 32)

    def test_gpu_embedding_model_config(self):
        """Test GPU embedding model with powers-of-2 range [1, 2, 4, 8]."""
        model = MagicMock()
        model.model_config_parameters.return_value = {
            "instance_group": [[{"kind": "KIND_GPU", "count": [1, 2, 4, 8]}]]
        }
        model.supports_batching.return_value = True

        dims = RunConfigGeneratorFactory._get_dimensions_for_model(model)

        # Verify instance_count dimension
        instance_dim = [d for d in dims if d.get_name() == "instance_count"][0]
        self.assertEqual(instance_dim.get_min_idx(), 0)  # 2^0 = 1
        self.assertEqual(instance_dim.get_max_idx(), 3)  # 2^3 = 8

        # Verify it produces the correct values
        self.assertEqual(instance_dim.get_value_at_idx(0), 1)
        self.assertEqual(instance_dim.get_value_at_idx(1), 2)
        self.assertEqual(instance_dim.get_value_at_idx(2), 4)
        self.assertEqual(instance_dim.get_value_at_idx(3), 8)


if __name__ == "__main__":
    unittest.main()
