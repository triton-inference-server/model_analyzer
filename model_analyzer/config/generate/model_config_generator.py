# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .config_generator_interface import ConfigGeneratorInterface
from .generator_utils import GeneratorUtils

from model_analyzer.triton.model.model_config import ModelConfig


class ModelConfigGenerator(ConfigGeneratorInterface):
    """ Given a model, generates model configs """

    def __init__(self, config, model, client):
        """
        Parameters
        ----------
        config: ModelAnalyzerConfig
        model: The model to generate ModelConfigs for
        client: TritonClient
        """
        self._client = client
        self._model_repository = config.model_repository
        self._num_retries = config.client_max_retries
        self._max_instance_count = config.run_config_search_max_instance_count
        self._search_disabled = config.run_config_search_disable
        self._remote_mode = config.triton_launch_mode == 'remote'
        self._base_model = model
        self._base_model_name = model.model_name()
        self._model_name_index = 0
        self._configs = self._generate_model_configs()

    def is_done(self):
        """ Returns true if this generator is done generating configs """
        return len(self._configs) == 0

    def next_config(self):
        """
        Returns
        -------
        ModelConfig
            The next ModelConfig generated by this class
        """
        return self._configs.pop(0)

    def set_last_results(self, measurement):
        """ 
        Given the results from the last ModelConfig, make decisions 
        about future configurations to generate

        Parameters
        ----------
        measurement: Measurement from the last run
        """
        pass

    def _generate_model_configs(self):
        """ Generate all model config combinations """
        if self._remote_mode:
            configs = self._get_remote_mode_model_configs()
        else:
            configs = self._get_direct_modes_model_configs()

        self._finalize_configs(configs)

        return configs

    def _get_remote_mode_model_configs(self):
        """ Generate model configs for remote mode """
        return [self._make_remote_model_config()]

    def _get_direct_modes_model_configs(self):
        """ Generate model configs for direct (non-remote) modes """
        model_configs = []
        param_combos = self._get_param_combinations()
        base_model_config = ModelConfig.create_from_file(
            f'{self._model_repository}/{self._base_model_name}')
        for param_combo in param_combos:
            model_config = self._make_direct_mode_model_config(
                base_model_config, param_combo)
            model_configs.append(model_config)

        return model_configs

    def _make_direct_mode_model_config(self, base_model_config, param_combo):
        """ 
        Given a base model config and a combination of parameters to change,
        apply the changes on top of the base and return the new model config
        """
        model_config_dict = base_model_config.get_config()
        if param_combo is not None:
            for key, value in param_combo.items():
                if value is not None:
                    model_config_dict[key] = value
        model_config = ModelConfig.create_from_dictionary(model_config_dict)
        return model_config

    def _get_param_combinations(self):
        """
        Calculate all parameter combinations to apply on top of the
        base model config
        """
        model_config_params = self._base_model.model_config_parameters()
        if model_config_params:
            param_combos = GeneratorUtils.generate_combinations(
                model_config_params)
        else:
            if self._search_disabled:
                param_combos = self._automatic_search_disabled_configs()
            else:
                param_combos = self._automatic_search_configs()

        if not self._is_default_config_in_configs(param_combos):
            self._add_default_config(param_combos)

        return param_combos

    def _automatic_search_disabled_configs(self):
        """ Return the configs when we want to search but searching is disabled """
        return [{}]

    def _automatic_search_configs(self):
        """ Search through automatic search variables to generate configs """

        configs = []
        kind = "KIND_GPU"
        if self._base_model.cpu_only():
            kind = "KIND_CPU"

        for instances in range(1, self._max_instance_count + 1):
            config = {
                'dynamic_batching': {},
                'instance_group': [{
                    'count': instances,
                    'kind': kind
                }]
            }
            configs.append(config)

        return configs

    def _make_remote_model_config(self):
        model_config = ModelConfig.create_from_triton_api(
            self._client, self._base_model_name, self._num_retries)
        return model_config

    def _is_default_config_in_configs(self, configs):
        return {} in configs

    def _add_default_config(self, configs):
        # Add in an empty configuration, which will apply the default values
        configs.append({})

    def _finalize_configs(self, configs):
        for config in configs:
            model_tmp_name = self._get_model_variant_name()
            config.set_field('name', model_tmp_name)
            config.set_cpu_only(self._base_model.cpu_only())

    def _get_model_variant_name(self):
        self._model_name_index += 1
        return f'{self._base_model_name}_config{self._model_name_index}'
