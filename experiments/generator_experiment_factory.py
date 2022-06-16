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

from model_analyzer.config.generate.run_config_generator import RunConfigGenerator
from model_analyzer.config.generate.undirected_run_config_generator import UndirectedRunConfigGenerator
from model_analyzer.config.generate.search_config import SearchConfig
from model_analyzer.config.generate.search_dimension import SearchDimension
from model_analyzer.config.generate.search_dimensions import SearchDimensions
from unittest.mock import MagicMock, patch


class GeneratorExperimentFactory:

    @staticmethod
    def create_generator(generator_name, config_command):
        """ 
        Create and return a RunConfig generator of the requested name
        
        As a side-effect, some patching may occur to allow the generator
        to run offline (without Perf Analyzer)

        Parameters
        ----------
        generator_name : string
            Name of the generator class to create
        config_command : ConfigCommandExperiment
            The config for model analyzer algorithm experiment
        """

        if generator_name == "RunConfigGenerator":
            generator = RunConfigGenerator(config_command,
                                           config_command.profile_models,
                                           MagicMock())
            p = patch(
                'model_analyzer.config.generate.run_config_generator.RunConfigGenerator.determine_triton_server_env'
            )
            p.start()

            return generator
        elif generator_name == "UndirectedRunConfigGenerator":
            dimensions = SearchDimensions()

            #yapf: disable
            for i, _ in enumerate(config_command.profile_models):
                dimensions.add(i, [
                    SearchDimension(f"max_batch_size", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
                    SearchDimension(f"instance_count", SearchDimension.DIMENSION_TYPE_LINEAR),
                    SearchDimension(f"concurrency", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
                ])
            #yapf: enable

            search_config = SearchConfig(
                dimensions=dimensions,
                radius=config_command.radius,
                step_magnitude=config_command.magnitude,
                min_initialized=config_command.min_initialized)

            generator = UndirectedRunConfigGenerator(
                search_config, config_command, config_command.profile_models)
            return generator
        else:
            raise Exception(f"Unknown generator {generator_name}")
