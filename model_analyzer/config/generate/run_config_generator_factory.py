#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Dict, List

from model_analyzer.config.generate.model_profile_spec import ModelProfileSpec
from model_analyzer.config.generate.model_variant_name_manager import (
    ModelVariantNameManager,
)
from model_analyzer.config.generate.search_parameters import SearchParameters
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.objects.config_model_profile_spec import (
    ConfigModelProfileSpec,
)
from model_analyzer.constants import LOGGER_NAME, MIN_INITIALIZED, RADIUS
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.triton.model.model_config import ModelConfig

from .brute_plus_binary_parameter_search_run_config_generator import (
    BrutePlusBinaryParameterSearchRunConfigGenerator,
)
from .config_generator_interface import ConfigGeneratorInterface
from .optuna_plus_concurrency_sweep_run_config_generator import (
    OptunaPlusConcurrencySweepRunConfigGenerator,
)
from .quick_plus_concurrency_sweep_run_config_generator import (
    QuickPlusConcurrencySweepRunConfigGenerator,
)
from .search_config import SearchConfig
from .search_dimension import SearchDimension
from .search_dimensions import SearchDimensions

logger = logging.getLogger(LOGGER_NAME)


class RunConfigGeneratorFactory:
    """
    Factory that creates the correct RunConfig Generators
    """

    @staticmethod
    def create_run_config_generator(
        command_config: ConfigCommandProfile,
        state_manager: AnalyzerStateManager,
        gpus: List[GPUDevice],
        models: List[ConfigModelProfileSpec],
        client: TritonClient,
        result_manager: ResultManager,
        model_variant_name_manager: ModelVariantNameManager,
        search_parameters: Dict[str, SearchParameters],
        composing_search_parameters: Dict[str, SearchParameters],
    ) -> ConfigGeneratorInterface:
        """
        Parameters
        ----------
        command_config: ConfigCommandProfile
            The Model Analyzer config file for the profile step
        state_manager: AnalyzerStateManager
            The object that allows control and update of checkpoint state
        gpus: List of GPUDevices
        models: list of ConfigModelProfileSpec
            The models to generate RunConfigs for
        client: TritonClient
            The client handle used to send requests to Triton
        result_manager: ResultManager
            The object that handles storing and sorting the results from the perf analyzer
        model_variant_name_manager: ModelVariantNameManager
            Maps model variants to config names
        search_parameters: SearchParameters
            The object that handles the users configuration search parameters
        composing_search_parameters: SearchParameters
            The object that handles the users configuration search parameters for composing models

        Returns
        -------
        A generator that implements ConfigGeneratorInterface and creates RunConfigs
        """

        new_models = []
        for model in models:
            new_models.append(ModelProfileSpec(model, command_config, client, gpus))

        composing_models = RunConfigGeneratorFactory._create_composing_models(
            new_models, command_config, client, gpus
        )

        for composing_model in composing_models:
            composing_search_parameters[
                composing_model.model_name()
            ] = SearchParameters(
                config=command_config,
                model=composing_model,
                is_composing_model=True,
            )

        if command_config.run_config_search_mode == "optuna":
            return RunConfigGeneratorFactory._create_optuna_plus_concurrency_sweep_run_config_generator(
                command_config=command_config,
                state_manager=state_manager,
                gpu_count=len(gpus),
                models=new_models,
                composing_models=composing_models,
                result_manager=result_manager,
                search_parameters=search_parameters,
                composing_search_parameters=composing_search_parameters,
                model_variant_name_manager=model_variant_name_manager,
            )
        elif command_config.run_config_search_mode == "quick" or composing_models:
            return RunConfigGeneratorFactory._create_quick_plus_concurrency_sweep_run_config_generator(
                command_config=command_config,
                gpu_count=len(gpus),
                models=new_models,
                composing_models=composing_models,
                result_manager=result_manager,
                model_variant_name_manager=model_variant_name_manager,
            )
        elif command_config.run_config_search_mode == "brute":
            return RunConfigGeneratorFactory._create_brute_plus_binary_parameter_search_run_config_generator(
                command_config=command_config,
                gpus=gpus,
                models=new_models,
                client=client,
                result_manager=result_manager,
                model_variant_name_manager=model_variant_name_manager,
            )
        else:
            raise TritonModelAnalyzerException(
                f"Unexpected search mode {command_config.run_config_search_mode}"
            )

    @staticmethod
    def _create_brute_plus_binary_parameter_search_run_config_generator(
        command_config: ConfigCommandProfile,
        gpus: List[GPUDevice],
        models: List[ModelProfileSpec],
        client: TritonClient,
        result_manager: ResultManager,
        model_variant_name_manager: ModelVariantNameManager,
    ) -> ConfigGeneratorInterface:
        return BrutePlusBinaryParameterSearchRunConfigGenerator(
            config=command_config,
            gpus=gpus,
            models=models,
            client=client,
            result_manager=result_manager,
            model_variant_name_manager=model_variant_name_manager,
        )

    @staticmethod
    def _create_optuna_plus_concurrency_sweep_run_config_generator(
        command_config: ConfigCommandProfile,
        state_manager: AnalyzerStateManager,
        gpu_count: int,
        models: List[ModelProfileSpec],
        composing_models: List[ModelProfileSpec],
        result_manager: ResultManager,
        model_variant_name_manager: ModelVariantNameManager,
        search_parameters: Dict[str, SearchParameters],
        composing_search_parameters: Dict[str, SearchParameters],
    ) -> ConfigGeneratorInterface:
        return OptunaPlusConcurrencySweepRunConfigGenerator(
            config=command_config,
            state_manager=state_manager,
            gpu_count=gpu_count,
            composing_models=composing_models,
            models=models,
            result_manager=result_manager,
            model_variant_name_manager=model_variant_name_manager,
            search_parameters=search_parameters,
            composing_search_parameters=composing_search_parameters,
        )

    @staticmethod
    def _create_quick_plus_concurrency_sweep_run_config_generator(
        command_config: ConfigCommandProfile,
        gpu_count: int,
        models: List[ModelProfileSpec],
        composing_models: List[ModelProfileSpec],
        result_manager: ResultManager,
        model_variant_name_manager: ModelVariantNameManager,
    ) -> ConfigGeneratorInterface:
        search_config = RunConfigGeneratorFactory._create_search_config(
            models, composing_models
        )
        return QuickPlusConcurrencySweepRunConfigGenerator(
            search_config=search_config,
            config=command_config,
            gpu_count=gpu_count,
            models=models,
            composing_models=composing_models,
            result_manager=result_manager,
            model_variant_name_manager=model_variant_name_manager,
        )

    @staticmethod
    def _create_search_config(
        models: List[ModelProfileSpec], composing_models: List[ModelProfileSpec]
    ) -> SearchConfig:
        dimensions = SearchDimensions()

        index = 0
        all_models = models + composing_models

        for model in all_models:
            # Top level ensemble models don't have any dimensions
            if model.is_ensemble():
                continue

            dims = RunConfigGeneratorFactory._get_dimensions_for_model(model)
            dimensions.add_dimensions(index, dims)
            index += 1

        search_config = SearchConfig(
            dimensions=dimensions, radius=RADIUS, min_initialized=MIN_INITIALIZED
        )

        return search_config

    @staticmethod
    def _get_dimensions_for_model(model: ModelProfileSpec) -> List[SearchDimension]:
        """
        Create search dimensions for a model, respecting user-specified
        instance_group count lists if provided.
        """
        dims = []

        # Check if user specified instance_group with a count list
        instance_count_list = RunConfigGeneratorFactory._get_instance_count_list(model)

        if instance_count_list:
            # User specified a list - create constrained dimension
            dim = RunConfigGeneratorFactory._create_instance_dimension_from_list(
                instance_count_list
            )
            dims.append(dim)
        else:
            # Use default unbounded dimension
            dims.append(
                SearchDimension("instance_count", SearchDimension.DIMENSION_TYPE_LINEAR)
            )

        # Add max_batch_size dimension if model supports batching
        if model.supports_batching():
            # For now, max_batch_size always uses default exponential dimension
            # Could be extended to support user-specified lists in the future
            dims.insert(
                0,
                SearchDimension(
                    "max_batch_size", SearchDimension.DIMENSION_TYPE_EXPONENTIAL
                ),
            )

        return dims

    @staticmethod
    def _get_instance_count_list(model: ModelProfileSpec) -> List[int]:
        """
        Extract instance_group count list from model config parameters if specified.

        Returns empty list if not specified or not a list.
        """
        model_config_params = model.model_config_parameters()
        if not model_config_params:
            return []

        if "instance_group" not in model_config_params:
            return []

        # instance_group structure: [[ {'kind': 'KIND_GPU', 'count': [1, 2, 4]} ]]
        # The outer lists are from config parsing wrapping
        instance_group = model_config_params["instance_group"]

        if not instance_group or not isinstance(instance_group, list):
            return []

        # Unwrap the nested structure
        if len(instance_group) > 0 and isinstance(instance_group[0], list):
            instance_group = instance_group[0]

        if len(instance_group) == 0 or not isinstance(instance_group[0], dict):
            return []

        count = instance_group[0].get("count")
        if isinstance(count, list) and len(count) > 0:
            return count

        return []

    @staticmethod
    def _create_instance_dimension_from_list(
        count_list: List[int],
    ) -> SearchDimension:
        """
        Create a SearchDimension for instance_count from a user-specified list.

        For lists that are powers of 2 (e.g., [1, 2, 4, 8, 16, 32]),
        uses EXPONENTIAL dimension type with appropriate min/max indexes.

        For other lists, uses LINEAR dimension type with appropriate min/max.

        Raises TritonModelAnalyzerException if the list is not compatible with
        either LINEAR or EXPONENTIAL growth patterns.
        """
        if not count_list or len(count_list) == 0:
            raise TritonModelAnalyzerException("Instance count list cannot be empty")

        # Sort the list to check for patterns
        sorted_counts = sorted(count_list)

        # Check if it's powers of 2
        if RunConfigGeneratorFactory._is_powers_of_two(sorted_counts):
            # Use EXPONENTIAL: 2^idx gives the value
            # For [1, 2, 4, 8, 16, 32]: min_idx=0 (2^0=1), max_idx=5 (2^5=32)
            min_idx = int(math.log2(sorted_counts[0]))
            max_idx = int(math.log2(sorted_counts[-1]))

            return SearchDimension(
                "instance_count",
                SearchDimension.DIMENSION_TYPE_EXPONENTIAL,
                min=min_idx,
                max=max_idx,
            )

        # Check if it's a contiguous linear sequence
        elif RunConfigGeneratorFactory._is_linear_sequence(sorted_counts):
            # Use LINEAR: idx+1 gives the value (LINEAR starts at 1, not 0)
            # For [1, 2, 3, 4]: min_idx=0 (0+1=1), max_idx=3 (3+1=4)
            min_idx = sorted_counts[0] - 1
            max_idx = sorted_counts[-1] - 1

            return SearchDimension(
                "instance_count",
                SearchDimension.DIMENSION_TYPE_LINEAR,
                min=min_idx,
                max=max_idx,
            )

        else:
            # List is not compatible with LINEAR or EXPONENTIAL
            raise TritonModelAnalyzerException(
                f"Instance count list {count_list} is not compatible with Quick search mode. "
                f"Lists must be either powers of 2 (e.g., [1, 2, 4, 8, 16, 32]) "
                f"or a contiguous sequence (e.g., [1, 2, 3, 4, 5])."
            )

    @staticmethod
    def _is_powers_of_two(sorted_list: List[int]) -> bool:
        """Check if all values in the list are powers of 2 and form a valid sequence."""
        for val in sorted_list:
            if val <= 0:
                return False
            # Check if val is a power of 2: log2(val) should be an integer
            log_val = math.log2(val)
            if not log_val.is_integer():
                return False

        return True

    @staticmethod
    def _is_linear_sequence(sorted_list: List[int]) -> bool:
        """Check if the list is a contiguous linear sequence."""
        if len(sorted_list) < 2:
            return True

        # Check if values are consecutive: diff should always be 1
        for i in range(1, len(sorted_list)):
            if sorted_list[i] - sorted_list[i - 1] != 1:
                return False

        return True

    @staticmethod
    def _get_batching_supported_dimensions() -> List[SearchDimension]:
        """Legacy method - kept for backward compatibility."""
        return [
            SearchDimension(
                f"max_batch_size", SearchDimension.DIMENSION_TYPE_EXPONENTIAL
            ),
            SearchDimension(f"instance_count", SearchDimension.DIMENSION_TYPE_LINEAR),
        ]

    @staticmethod
    def _get_batching_not_supported_dimensions() -> List[SearchDimension]:
        """Legacy method - kept for backward compatibility."""
        return [
            SearchDimension(f"instance_count", SearchDimension.DIMENSION_TYPE_LINEAR)
        ]

    @staticmethod
    def _create_composing_models(
        models: List[ModelProfileSpec],
        config: ConfigCommandProfile,
        client: TritonClient,
        gpus: List[GPUDevice],
    ) -> List[ModelProfileSpec]:
        """
        Given a list of models create a list of all the composing models (BLS + Ensemble)
        """
        composing_models = RunConfigGeneratorFactory._create_bls_composing_models(
            config, client, gpus
        )

        for model in models:
            composing_models.extend(
                RunConfigGeneratorFactory._create_ensemble_composing_models(
                    model, config, client, gpus
                )
            )

        for composing_model in composing_models:
            if composing_model.is_ensemble():
                raise TritonModelAnalyzerException(
                    f"Model Analyzer does not support ensembles as a composing model type: {composing_model.model_name()}"
                )

        return composing_models

    @staticmethod
    def _create_bls_composing_models(
        config: ConfigCommandProfile, client: TritonClient, gpus: List[GPUDevice]
    ) -> List[ModelProfileSpec]:
        """
        Creates a list of BLS composing model configs based on the profile command config
        """
        bls_composing_model_configs = [
            ModelProfileSpec(bls_composing_model_spec, config, client, gpus)
            for bls_composing_model_spec in config.bls_composing_models
        ]

        return bls_composing_model_configs

    @staticmethod
    def _create_ensemble_composing_models(
        model: ModelProfileSpec,
        config: ConfigCommandProfile,
        client: TritonClient,
        gpus: List[GPUDevice],
    ) -> List[ModelProfileSpec]:
        """
        Creates a list of Ensemble composing model configs based on the model.

        If user specified ensemble_composing_models configs, use those for matching models.
        Otherwise, use auto-discovered configs from ensemble_scheduling.
        """
        model_config = ModelConfig.create_from_profile_spec(model, config, client, gpus)

        if not model_config.is_ensemble():
            return []

        # Auto-discover composing model names from ensemble_scheduling
        ensemble_composing_model_names = model_config.get_ensemble_composing_models()
        if ensemble_composing_model_names is None:
            return []

        # Check if user provided configs for any of these models
        user_provided_configs = {}
        if config.ensemble_composing_models is not None:
            for user_spec in config.ensemble_composing_models:
                user_provided_configs[user_spec.model_name()] = user_spec

        # Create ModelProfileSpecs, using user configs when available
        ensemble_composing_model_configs = []
        for model_name in ensemble_composing_model_names:
            if model_name in user_provided_configs:
                # Use user-provided config with model_config_parameters
                model_spec = user_provided_configs[model_name]
            else:
                # Use auto-discovered config (just model name, no parameters)
                model_spec = ConfigModelProfileSpec(model_name)

            mps = ModelProfileSpec(model_spec, config, client, gpus)
            ensemble_composing_model_configs.append(mps)

        # Warn if user specified models that aren't in the ensemble
        if user_provided_configs:
            unused_models = set(user_provided_configs.keys()) - set(
                ensemble_composing_model_names
            )
            if unused_models:
                logger.warning(
                    f"The following models in ensemble_composing_models were not found "
                    f"in the ensemble '{model.model_name()}' and will be ignored: "
                    f"{', '.join(sorted(unused_models))}"
                )

        return ensemble_composing_model_configs
