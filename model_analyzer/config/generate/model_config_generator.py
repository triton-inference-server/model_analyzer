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

from model_analyzer.config.generate.generator_utils import GeneratorUtils


class ModelConfigGenerator:
    """ Given a model, generates model configs """

    def __init__(self, config, model):
        self._max_instance_count = config.run_config_search_max_instance_count
        self._search_disabled = config.run_config_search_disable
        self._remote_mode = config.triton_launch_mode == 'remote'
        self._model = model
        self._configs = self._generate_model_configs()

    def is_done(self):
        """ Returns true if this generator is done generating configs """
        return len(self._configs) == 0

    def next_config(self):
        """ Returns the next generated config """
        return self._configs.pop(0)

    def _generate_model_configs(self):
        """ Generate all model config combinations """
        if self._remote_mode:
            configs = self._get_remote_mode_model_configs()
        else:
            configs = self._get_local_mode_model_configs()

        if not self._is_default_config_in_configs(configs):
            self._add_default_config(configs)

        return configs

    def _get_remote_mode_model_configs(self):
        """ Generate model configs for remote mode """
        return []

    def _get_local_mode_model_configs(self):
        """ Generate model configs for local mode """

        model_config_params = self._model.model_config_parameters()
        if model_config_params:
            configs = GeneratorUtils.generate_combinations(model_config_params)
        else:
            configs = self._automatic_search_configs()

        return configs

    def _automatic_search_configs(self):
        """ Search through automatic search variables to generate configs """

        configs = []
        if not self._search_disabled:
            for instances in range(1, self._max_instance_count + 1):
                config = {'dynamic_batching': {}, 'instance_count': instances}
                configs.append(config)

        return configs

    def _is_default_config_in_configs(self, configs):
        return None in configs

    def _add_default_config(self, configs):
        # Add in an empty configuration, which will apply the default values
        configs.append(None)
