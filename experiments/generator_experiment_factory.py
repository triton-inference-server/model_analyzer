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
from unittest.mock import MagicMock, patch


class GeneratorExperimentFactory:

    @staticmethod
    def create_generator(generator_name, config_command):
        """ 
        Given a generator name, create and return it

        As a side-effect, some patching may occur to allow the generator
        to run in an offline scenario.
        """

        if generator_name == "RunConfigGenerator":
            generator = RunConfigGenerator(config_command,
                                           config_command.profile_models,
                                           MagicMock())
            p = patch(
                'model_analyzer.config.generate.run_config_generator.RunConfigGenerator._determine_triton_server_env'
            )
            p.start()

            return generator
        elif generator_name == "UndirectedRunConfigGenerator":
            generator = UndirectedRunConfigGenerator(
                config_command, config_command.profile_models, MagicMock())
            return generator
        else:
            raise Exception(f"Unknown generator {generator_name}")
