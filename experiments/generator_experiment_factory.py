#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import MagicMock, patch

from model_analyzer.config.generate.model_variant_name_manager import (
    ModelVariantNameManager,
)
from model_analyzer.config.generate.run_config_generator_factory import (
    RunConfigGeneratorFactory,
)
from model_analyzer.config.generate.search_dimension import SearchDimension


class GeneratorExperimentFactory:
    command_config = None

    @staticmethod
    def create_generator(config_command):
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

        GeneratorExperimentFactory.config_command = config_command

        p1 = patch(
            "model_analyzer.config.generate.run_config_generator_factory.RunConfigGeneratorFactory._get_batching_supported_dimensions",
            GeneratorExperimentFactory.get_batching_supported_dimensions,
        )
        p2 = patch(
            "model_analyzer.config.generate.run_config_generator_factory.RunConfigGeneratorFactory._get_batching_not_supported_dimensions",
            GeneratorExperimentFactory.get_batching_not_supported_dimensions,
        )
        p1.start()
        p2.start()
        mvn = ModelVariantNameManager()
        generator = RunConfigGeneratorFactory.create_run_config_generator(
            config_command,
            MagicMock(),
            config_command.profile_models,
            MagicMock(),
            MagicMock(),
            mvn,
        )
        return generator

    @staticmethod
    def get_batching_supported_dimensions():
        mbs_min = GeneratorExperimentFactory.config_command.min_mbs_index
        ret = [
            SearchDimension(
                f"max_batch_size", SearchDimension.DIMENSION_TYPE_EXPONENTIAL, mbs_min
            )
        ]
        if GeneratorExperimentFactory.config_command.exponential_inst_count:
            ret.append(
                SearchDimension(
                    f"instance_count", SearchDimension.DIMENSION_TYPE_EXPONENTIAL
                )
            )
        else:
            ret.append(
                SearchDimension(
                    f"instance_count", SearchDimension.DIMENSION_TYPE_LINEAR
                )
            )
        return ret

    @staticmethod
    def get_batching_not_supported_dimensions():
        mbs_min = GeneratorExperimentFactory.config_command.min_mbs_index

        ret = [
            SearchDimension(
                f"concurrency", SearchDimension.DIMENSION_TYPE_EXPONENTIAL, mbs_min
            )
        ]
        if GeneratorExperimentFactory.config_command.exponential_inst_count:
            ret.append(
                SearchDimension(
                    f"instance_count", SearchDimension.DIMENSION_TYPE_EXPONENTIAL
                )
            )
        else:
            ret.append(
                SearchDimension(
                    f"instance_count", SearchDimension.DIMENSION_TYPE_LINEAR
                )
            )
        return ret
