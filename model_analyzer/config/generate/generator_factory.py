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

from .automatic_model_config_generator import AutomaticModelConfigGenerator
from .manual_model_config_generator import ManualModelConfigGenerator


class ConfigGeneratorFactory:
    """
    Factory that creates the correct Config Generators
    """

    @staticmethod
    def create_model_config_generator(config, gpus, model, client,
                                      variant_name_manager, default_only,
                                      early_exit_enable):
        """
        Parameters
        ----------
        config: ConfigCommandProfile
            The Model Analyzer config file for the profile step
        gpus: List of GPUDevices
        model: ConfigModelProfileSpec
            The model to generate ModelRunConfigs for
        client: TritonClient
            The client handle used to send requests to Triton
        variant_name_manager: ModelVariantNameManager
            Used to manage the model variant names
        default_only: Bool 
            If true, only the default config will be generated by the created generator
            If false, the default config will NOT be generated by the created generator
        early_exit_enable: Bool
            If true, the created generator can early exit if throughput plateaus

        Returns
        -------
        ModelConfigGenerator
        """

        remote_mode = config.triton_launch_mode == 'remote'
        search_disabled = config.run_config_search_disable
        model_config_params = model.model_config_parameters()

        if (remote_mode or search_disabled or model_config_params):
            return ManualModelConfigGenerator(config, gpus, model, client,
                                              variant_name_manager,
                                              default_only, early_exit_enable)
        else:
            return AutomaticModelConfigGenerator(config, gpus, model, client,
                                                 variant_name_manager,
                                                 default_only,
                                                 early_exit_enable)
