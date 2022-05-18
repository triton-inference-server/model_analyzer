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
from unittest.mock import MagicMock, patch


class GeneratorExperimentFactory:

    @staticmethod
    def get_generator_and_patches(generator_name, profile_config):
        """ 
        Given a generator name, create that and return that generator along
        with a list of patches to apply when using the generator
        """
        if generator_name == "RunConfigGenerator":
            generator = RunConfigGenerator(profile_config,
                                           profile_config.profile_models,
                                           MagicMock())
            patches = [
                patch(
                    'model_analyzer.config.generate.run_config_generator.RunConfigGenerator._determine_triton_server_env'
                )
            ]

            return [generator, patches]
        else:
            raise Exception(f"Unknown generator {generator_name}")
