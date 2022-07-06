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


class RunConfigGeneratorFactory:
    """
    Factory that creates the correct RunConfig Generators
    """

    @staticmethod
    def create_run_config_generator(command_config, gpus, models, client):
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

        Returns
        -------
        A generator that implements ConfigGeneratorInterface and creates RunConfigs        
        """

        if (command_config.run_config_search_mode == "quick"):
            return RunConfigGeneratorFactory.create_quick_run_config_generator(
                command_config=command_config,
                gpus=gpus,
                models=models,
                client=client)
        elif (command_config.run_config_search_mode == "brute"):
            return RunConfigGeneratorFactory._create_brute_run_config_generator(
                command_config=command_config,
                gpus=gpus,
                models=models,
                client=client)
        else:
            raise TritonModelAnalyzerException(
                f"Unexpected search mode {command_config.run_config_search_mode}"
            )

    @staticmethod
    def _create_brute_run_config_generator(command_config, gpus, models,
                                           client):
        return BruteRunConfigGenerator(config=command_config,
                                       gpus=gpus,
                                       models=models,
                                       client=client)

    @staticmethod
    def create_quick_run_config_generator(command_config, gpus, models, client):
        search_config = RunConfigGeneratorFactory._create_search_config(
            command_config)
        return QuickRunConfigGenerator(search_config=search_config,
                                       config=command_config,
                                       gpus=gpus,
                                       models=models,
                                       client=client)

    @staticmethod
    def _create_search_config(command_config):
        dimensions = SearchDimensions()

        #yapf: disable
        for i, _ in enumerate(command_config.profile_models):
            dimensions.add_dimensions(i, [
                SearchDimension(f"max_batch_size", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                SearchDimension(f"instance_count", SearchDimension.DIMENSION_TYPE_LINEAR),
                SearchDimension(f"concurrency", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
            ])
        #yapf: enable

        # TODO TMA-746: do we want to expose these options to CLI?
        search_config = SearchConfig(dimensions=dimensions,
                                     radius=2,
                                     step_magnitude=2,
                                     min_initialized=3)

        return search_config
