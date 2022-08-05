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

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from .brute_run_config_generator import BruteRunConfigGenerator
from .quick_run_config_generator import QuickRunConfigGenerator
from .search_dimensions import SearchDimensions
from .search_dimension import SearchDimension
from .search_config import SearchConfig

from model_analyzer.constants import RADIUS, MAGNITUDE, MIN_INITIALIZED


class RunConfigGeneratorFactory:
    """
    Factory that creates the correct RunConfig Generators
    """

    @staticmethod
    def create_run_config_generator(command_config, gpus, models, client,
                                    model_variant_name_manager):
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
        model_variant_name_manager: ModelVariantNameManager
            Maps model variants to config names

        Returns
        -------
        A generator that implements ConfigGeneratorInterface and creates RunConfigs        
        """

        if (command_config.run_config_search_mode == "quick"):
            return RunConfigGeneratorFactory.create_quick_run_config_generator(
                command_config=command_config,
                gpus=gpus,
                models=models,
                client=client,
                model_variant_name_manager=model_variant_name_manager)
        elif (command_config.run_config_search_mode == "brute"):
            return RunConfigGeneratorFactory._create_brute_run_config_generator(
                command_config=command_config,
                gpus=gpus,
                models=models,
                client=client,
                model_variant_name_manager=model_variant_name_manager)
        else:
            raise TritonModelAnalyzerException(
                f"Unexpected search mode {command_config.run_config_search_mode}"
            )

    @staticmethod
    def _create_brute_run_config_generator(command_config, gpus, models, client,
                                           model_variant_name_manager):
        return BruteRunConfigGenerator(
            config=command_config,
            gpus=gpus,
            models=models,
            client=client,
            model_variant_name_manager=model_variant_name_manager)

    @staticmethod
    def create_quick_run_config_generator(command_config, gpus, models, client,
                                          model_variant_name_manager):
        search_config = RunConfigGeneratorFactory._create_search_config(
            command_config)
        return QuickRunConfigGenerator(
            search_config=search_config,
            config=command_config,
            gpus=gpus,
            models=models,
            client=client,
            model_variant_name_manager=model_variant_name_manager)

    @staticmethod
    def _create_search_config(command_config):
        dimensions = SearchDimensions()

        #yapf: disable
        for i, _ in enumerate(command_config.profile_models):
            dimensions.add_dimensions(i, [
                SearchDimension(f"max_batch_size", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension(f"instance_count", SearchDimension.DIMENSION_TYPE_LINEAR)
            ])
        #yapf: enable

        search_config = SearchConfig(dimensions=dimensions,
                                     radius=RADIUS,
                                     step_magnitude=MAGNITUDE,
                                     min_initialized=MIN_INITIALIZED)

        return search_config
