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

from .base_model_config_generator import BaseModelConfigGenerator
from .generator_utils import GeneratorUtils

from model_analyzer.triton.model.model_config import ModelConfig

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class ManualModelConfigGenerator(BaseModelConfigGenerator):
    """ Given a model, generates model configs in manual search mode """

    def __init__(self, config, model, client):
        """
        Parameters
        ----------
        config: ModelAnalyzerConfig
        model: The model to generate ModelConfigs for
        client: TritonClient
        """
        super().__init__(config, model, client)

        self._reload_model_disable = config.reload_model_disable
        self._num_retries = config.client_max_retries
        self._search_disabled = config.run_config_search_disable
        self._curr_config_index = 0
        self._configs = self._generate_model_configs()

    def _done_walking(self):
        return len(self._configs) == self._curr_config_index + 1

    def _step(self):
        self._curr_config_index += 1

    def _get_next_model_config(self):
        return self._configs[self._curr_config_index]

    def _generate_model_configs(self):
        """ Generate all model config combinations """
        if self._remote_mode:
            configs = self._generate_remote_mode_model_configs()
        else:
            configs = self._generate_direct_modes_model_configs()

        return configs

    def _generate_remote_mode_model_configs(self):
        """ Generate model configs for remote mode """
        return [self._make_remote_model_config()]

    def _generate_direct_modes_model_configs(self):
        """ Generate model configs for direct (non-remote) modes """
        model_configs = []
        param_combos = self._get_param_combinations()
        for param_combo in param_combos:
            model_config = self._make_direct_mode_model_config(param_combo)
            model_config.set_cpu_only(self._cpu_only)

            model_configs.append(model_config)

        return model_configs

    def _get_param_combinations(self):
        """
        Calculate all parameter combinations to apply on top of the
        base model config for manual search 
        """
        model_config_params = self._base_model.model_config_parameters()
        if model_config_params:
            param_combos = GeneratorUtils.generate_combinations(
                model_config_params)
        else:
            if self._search_disabled:
                param_combos = self._generate_search_disabled_param_combos()
            else:
                raise TritonModelAnalyzerException(
                    f"Automatic search not supported in ManualModelConfigGenerator"
                )

        if not self._is_default_combo_in_param_combos(param_combos):
            self._add_default_combo(param_combos)

        return param_combos

    def _is_default_combo_in_param_combos(self, param_combos):
        return self.DEFAULT_PARAM_COMBO in param_combos

    def _add_default_combo(self, param_combos):
        # Add in an empty combo at the start of the list, which will just apply the default values
        #
        param_combos.insert(0, self.DEFAULT_PARAM_COMBO)

    def _generate_search_disabled_param_combos(self):
        """ Return the configs when we want to search but searching is disabled """
        return [self.DEFAULT_PARAM_COMBO]
