# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List

from model_analyzer.config.generate.model_variant_name_manager import ModelVariantNameManager
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.config.input.objects.config_model_profile_spec import ConfigModelProfileSpec
from model_analyzer.config.generate.model_profile_spec import ModelProfileSpec
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.result.result_manager import ResultManager
from .brute_run_config_generator import BruteRunConfigGenerator
from .quick_plus_concurrency_sweep_run_config_generator import QuickPlusConcurrencySweepRunConfigGenerator
from .brute_plus_binary_parameter_search_run_config_generator import BrutePlusBinaryParameterSearchRunConfigGenerator
from .search_dimensions import SearchDimensions
from .search_dimension import SearchDimension
from .search_config import SearchConfig
from typing import List
from model_analyzer.constants import RADIUS, MIN_INITIALIZED
from .config_generator_interface import ConfigGeneratorInterface


class RunConfigGeneratorFactory:
    """
    Factory that creates the correct RunConfig Generators
    """

    @staticmethod
    def create_run_config_generator(
        command_config: ConfigCommandProfile, gpus: List[GPUDevice],
        models: List[ConfigModelProfileSpec], client: TritonClient,
        result_manager: ResultManager,
        model_variant_name_manager: ModelVariantNameManager
    ) -> ConfigGeneratorInterface:
        """
        Parameters
        ----------
        command_config: ConfigCommandProfile
            The Model Analyzer config file for the profile step
        gpus: List of GPUDevices
        models: list of ConfigModelProfileSpec
            The models to generate RunConfigs for
        client: TritonClient
            The client handle used to send requests to Triton
        result_manager: ResultManager
            The object that handles storing and sorting the results from the perf analyzer
        model_variant_name_manager: ModelVariantNameManager
            Maps model variants to config names

        Returns
        -------
        A generator that implements ConfigGeneratorInterface and creates RunConfigs        
        """

        new_models = []
        for model in models:
            new_models.append(
                ModelProfileSpec(model, command_config, client, gpus))

        composing_models = RunConfigGeneratorFactory._create_composing_models(
            new_models, command_config, client, gpus)

        if (command_config.run_config_search_mode == "quick" or
                composing_models):
            return RunConfigGeneratorFactory._create_quick_plus_concurrency_sweep_run_config_generator(
                command_config=command_config,
                gpus=gpus,
                models=new_models,
                composing_models=composing_models,
                client=client,
                result_manager=result_manager,
                model_variant_name_manager=model_variant_name_manager)
        elif (command_config.run_config_search_mode == "brute"):
            return RunConfigGeneratorFactory._create_brute_plus_binary_parameter_search_run_config_generator(
                command_config=command_config,
                gpus=gpus,
                models=new_models,
                client=client,
                result_manager=result_manager,
                model_variant_name_manager=model_variant_name_manager)
        else:
            raise TritonModelAnalyzerException(
                f"Unexpected search mode {command_config.run_config_search_mode}"
            )

    @staticmethod
    def _create_brute_plus_binary_parameter_search_run_config_generator(
        command_config: ConfigCommandProfile, gpus: List[GPUDevice],
        models: List[ModelProfileSpec], client: TritonClient,
        result_manager: ResultManager,
        model_variant_name_manager: ModelVariantNameManager
    ) -> ConfigGeneratorInterface:
        return BrutePlusBinaryParameterSearchRunConfigGenerator(
            config=command_config,
            gpus=gpus,
            models=models,
            client=client,
            result_manager=result_manager,
            model_variant_name_manager=model_variant_name_manager)

    @staticmethod
    def _create_quick_plus_concurrency_sweep_run_config_generator(
        command_config: ConfigCommandProfile, gpus: List[GPUDevice],
        models: List[ModelProfileSpec],
        composing_models: List[ModelProfileSpec], client: TritonClient,
        result_manager: ResultManager,
        model_variant_name_manager: ModelVariantNameManager
    ) -> ConfigGeneratorInterface:
        search_config = RunConfigGeneratorFactory._create_search_config(
            models, composing_models)
        return QuickPlusConcurrencySweepRunConfigGenerator(
            search_config=search_config,
            config=command_config,
            gpus=gpus,
            models=models,
            composing_models=composing_models,
            client=client,
            result_manager=result_manager,
            model_variant_name_manager=model_variant_name_manager)

    @staticmethod
    def _create_search_config(
            models: List[ModelProfileSpec],
            composing_models: List[ModelProfileSpec]) -> SearchConfig:
        dimensions = SearchDimensions()

        index = 0
        all_models = models + composing_models

        for model in all_models:
            # Top level ensemble models don't have any dimensions
            if model.is_ensemble():
                continue

            dims = RunConfigGeneratorFactory._get_dimensions_for_model(
                model.supports_batching())
            dimensions.add_dimensions(index, dims)
            index += 1

        search_config = SearchConfig(dimensions=dimensions,
                                     radius=RADIUS,
                                     min_initialized=MIN_INITIALIZED)

        return search_config

    @staticmethod
    def _get_dimensions_for_model(
            is_batching_supported: bool) -> List[SearchDimension]:

        if (is_batching_supported):
            return RunConfigGeneratorFactory._get_batching_supported_dimensions(
            )
        else:
            return RunConfigGeneratorFactory._get_batching_not_supported_dimensions(
            )

    @staticmethod
    def _get_batching_supported_dimensions() -> List[SearchDimension]:
        return [
            SearchDimension(f"max_batch_size",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension(f"instance_count",
                            SearchDimension.DIMENSION_TYPE_LINEAR)
        ]

    @staticmethod
    def _get_batching_not_supported_dimensions() -> List[SearchDimension]:
        return [
            SearchDimension(f"instance_count",
                            SearchDimension.DIMENSION_TYPE_LINEAR)
        ]

    @staticmethod
    def _create_composing_models(
            models: List[ModelProfileSpec], config: ConfigCommandProfile,
            client: TritonClient,
            gpus: List[GPUDevice]) -> List[ModelProfileSpec]:
        """
        Given a list of models create a list of all the composing models (BLS + Ensemble)
        """
        composing_models = RunConfigGeneratorFactory._create_bls_composing_models(
            config, client, gpus)

        for model in models:
            composing_models.extend(
                RunConfigGeneratorFactory._create_ensemble_composing_models(
                    model, config, client, gpus))

        for composing_model in composing_models:
            if composing_model.is_ensemble():
                raise TritonModelAnalyzerException(
                    f"Model Analyzer does not support ensembles as a composing model type: {composing_model.model_name()}"
                )

        return composing_models

    @staticmethod
    def _create_bls_composing_models(
            config: ConfigCommandProfile, client: TritonClient,
            gpus: List[GPUDevice]) -> List[ModelProfileSpec]:
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
            model: ModelProfileSpec, config: ConfigCommandProfile,
            client: TritonClient,
            gpus: List[GPUDevice]) -> List[ModelProfileSpec]:
        """
        Creates a list of Ensemble composing model configs based on the model
        """
        model_config = ModelConfig.create_from_profile_spec(
            model, config, client, gpus)

        if not model_config.is_ensemble():
            return []

        ensemble_composing_model_names = model_config.get_ensemble_composing_models(
        )

        ensemble_composing_model_specs = ConfigModelProfileSpec.model_list_to_config_model_profile_spec(
            ensemble_composing_model_names)

        ensemble_composing_model_configs = [
            ModelProfileSpec(ensemble_composing_model_spec, config, client,
                             gpus)
            for ensemble_composing_model_spec in ensemble_composing_model_specs
        ]

        return ensemble_composing_model_configs
