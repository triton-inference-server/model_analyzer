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
from model_analyzer.constants import LOGGER_NAME, DEFAULT_CONFIG_PARAMS

from model_analyzer.triton.model.model_config import ModelConfig
import logging
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

logger = logging.getLogger(LOGGER_NAME)
from copy import deepcopy


class ManualModelConfigGenerator(BaseModelConfigGenerator):
    """ Given a model, generates model configs in manual search mode """

    def __init__(self, config, model, client, variant_name_manager,
                 default_only, early_exit_enable):
        """
        Parameters
        ----------
        config: ModelAnalyzerConfig
        model: The model to generate ModelConfigs for
        client: TritonClient
        variant_name_manager: ModelVariantNameManager
        default_only: Bool 
            If true, only the default config will be generated
            If false, the default config will NOT be generated                
        early_exit_enable: Bool
            If true, the generator can early exit if throughput plateaus
        """
        super().__init__(config, model, client, variant_name_manager,
                         default_only, early_exit_enable)

        self._reload_model_disable = config.reload_model_disable
        self._num_retries = config.client_max_retries
        self._search_disabled = config.run_config_search_disable
        self._curr_config_index = 0
        self._curr_max_batch_size_index = 0

        self._max_batch_sizes = None
        self._non_max_batch_size_param_combos = []
        self._determine_max_batch_sizes_and_param_combos()

        # All configs are pregenerated in _configs[][]
        # Indexed as follows:
        #    _configs[_curr_config_index][_curr_max_batch_size_index]
        #
        self._configs = self._generate_model_configs()

    def _done_walking(self):
        return self._done_walking_configs() \
           and self._done_walking_max_batch_size()

    def _done_walking_configs(self):
        return len(self._configs) == self._curr_config_index + 1

    def _done_walking_max_batch_size(self):
        if (self._max_batch_sizes is None or len(self._max_batch_sizes)
                == self._curr_max_batch_size_index + 1):
            return True

        if self._early_exit_enable and not self._last_results_increased_throughput(
        ):
            self._print_max_batch_size_plateau_warning()
            return True
        return False

    def _step(self):
        if self._done_walking_max_batch_size():
            self._reset_max_batch_size()
            self._step_config()
        else:
            self._step_max_batch_size()

    def _reset_max_batch_size(self):
        super()._reset_max_batch_size()
        self._curr_max_batch_size_index = 0

    def _step_config(self):
        self._curr_config_index += 1

    def _step_max_batch_size(self):
        self._curr_max_batch_size_index += 1

        last_max_throughput = self._get_last_results_max_throughput()
        self._curr_max_batch_size_throughputs.append(last_max_throughput)

    def _get_next_model_config(self):
        return self._configs[self._curr_config_index][
            self._curr_max_batch_size_index]

    def _generate_model_configs(self):
        """ Generate all model config combinations """
        if self._remote_mode:
            configs = self._generate_remote_mode_model_configs()
        else:
            configs = self._generate_direct_modes_model_configs()

        return configs

    def _generate_remote_mode_model_configs(self):
        """ Generate model configs for remote mode """
        return [[self._make_remote_model_config()]]

    def _generate_direct_modes_model_configs(self):
        """ Generate model configs for direct (non-remote) modes """
        model_configs = []
        for param_combo in self._non_max_batch_size_param_combos:
            configs_with_max_batch_size = []
            if self._max_batch_sizes:
                for mbs in self._max_batch_sizes:
                    param_combo['max_batch_size'] = mbs
                    model_config = self._make_direct_mode_model_config(
                        param_combo)
                    configs_with_max_batch_size.append(model_config)
            else:
                model_config = self._make_direct_mode_model_config(param_combo)
                configs_with_max_batch_size.append(model_config)

            model_configs.append(configs_with_max_batch_size)

        return model_configs

    def _determine_max_batch_sizes_and_param_combos(self):
        """
        Determine self._max_batch_sizes and self._non_max_batch_size_param_combos 
        """
        if self._remote_mode:
            return

        if self._default_only:
            self._non_max_batch_size_param_combos = [DEFAULT_CONFIG_PARAMS]
        else:
            model_config_params = deepcopy(
                self._base_model.model_config_parameters())
            if model_config_params:
                self._max_batch_sizes = model_config_params.pop(
                    "max_batch_size", None)
                self._non_max_batch_size_param_combos = GeneratorUtils.generate_combinations(
                    model_config_params)
            else:
                if self._search_disabled:
                    self._non_max_batch_size_param_combos = self._generate_search_disabled_param_combos(
                    )
                else:
                    raise TritonModelAnalyzerException(
                        f"Automatic search not supported in ManualModelConfigGenerator"
                    )

    def _generate_search_disabled_param_combos(self):
        """ Return the configs when we want to search but searching is disabled """
        return [DEFAULT_CONFIG_PARAMS]
