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

import logging
from typing import Dict, List

from model_analyzer.config.generate.model_variant_name_manager import (
    ModelVariantNameManager,
)
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.constants import DEFAULT_CONFIG_PARAMS, LOGGER_NAME
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.triton.model.model_config_variant import ModelConfigVariant

from .base_model_config_generator import BaseModelConfigGenerator
from .generator_utils import GeneratorUtils
from .model_profile_spec import ModelProfileSpec

logger = logging.getLogger(LOGGER_NAME)
from copy import deepcopy


class ManualModelConfigGenerator(BaseModelConfigGenerator):
    """Given a model, generates model configs in manual search mode"""

    _log_first_run = False

    def __init__(
        self,
        config: ConfigCommandProfile,
        gpus: List[GPUDevice],
        model: ModelProfileSpec,
        client: TritonClient,
        model_variant_name_manager: ModelVariantNameManager,
        default_only: bool,
        early_exit_enable: bool,
    ) -> None:
        """
        Parameters
        ----------
        config: ModelAnalyzerConfig
        gpus: List of GPUDevices
        model: The model to generate ModelConfigs for
        client: TritonClient
        model_variant_name_manager: ModelVariantNameManager
        default_only: Bool
            If true, only the default config will be generated
            If false, the default config will NOT be generated
        early_exit_enable: Bool
            If true, the generator can early exit if throughput plateaus
        """
        super().__init__(
            config,
            gpus,
            model,
            client,
            model_variant_name_manager,
            default_only,
            early_exit_enable,
        )

        if not ManualModelConfigGenerator._log_first_run:
            logger.info("")
            logger.info("Starting manual brute search")
            logger.info("")
            ManualModelConfigGenerator._log_first_run = True

        self._search_disabled = config.run_config_search_disable
        self._curr_config_index = 0
        self._curr_max_batch_size_index = 0

        self._max_batch_sizes = None
        self._non_max_batch_size_param_combos: List[Dict] = []
        self._determine_max_batch_sizes_and_param_combos()

        # All configs are pregenerated in _configs[][]
        # Indexed as follows:
        #    _configs[_curr_config_index][_curr_max_batch_size_index]
        #
        self._configs = self._generate_model_config_variants()

    def _done_walking(self) -> bool:
        return len(self._configs) == self._curr_config_index

    def _done_walking_max_batch_size(self) -> bool:
        if (
            self._max_batch_sizes is None
            or len(self._max_batch_sizes) == self._curr_max_batch_size_index
        ):
            return True

        if self._early_exit_enable and self._last_results_erroneous():
            return True

        if self._early_exit_enable and not self._last_results_increased_throughput():
            self._print_max_batch_size_plateau_warning()
            return True
        return False

    def _step(self) -> None:
        self._step_max_batch_size()

        if self._done_walking_max_batch_size():
            self._reset_max_batch_size()
            self._step_config()

    def _reset_max_batch_size(self) -> None:
        super()._reset_max_batch_size()
        self._curr_max_batch_size_index = 0

    def _step_config(self) -> None:
        self._curr_config_index += 1

    def _step_max_batch_size(self) -> None:
        self._curr_max_batch_size_index += 1

        last_max_throughput = self._get_last_results_max_throughput()
        if last_max_throughput:
            self._curr_max_batch_size_throughputs.append(last_max_throughput)

    def _get_next_model_config_variant(self) -> ModelConfigVariant:
        return self._configs[self._curr_config_index][self._curr_max_batch_size_index]

    def _generate_model_config_variants(self) -> List[List[ModelConfigVariant]]:
        """Generate all model config combinations"""

        model_config_variants = []
        for param_combo in self._non_max_batch_size_param_combos:
            configs_with_max_batch_size = []
            if self._max_batch_sizes:
                for mbs in self._max_batch_sizes:
                    param_combo["max_batch_size"] = mbs
                    model_config_variant = self._make_direct_mode_model_config_variant(
                        param_combo
                    )
                    configs_with_max_batch_size.append(model_config_variant)
            else:
                model_config_variant = self._make_direct_mode_model_config_variant(
                    param_combo
                )
                configs_with_max_batch_size.append(model_config_variant)

            model_config_variants.append(configs_with_max_batch_size)

        return model_config_variants

    def _determine_max_batch_sizes_and_param_combos(self) -> None:
        """
        Determine self._max_batch_sizes and self._non_max_batch_size_param_combos
        """
        if self._default_only:
            self._non_max_batch_size_param_combos = [DEFAULT_CONFIG_PARAMS]
        else:
            model_config_params = deepcopy(self._base_model.model_config_parameters())
            if model_config_params:
                self._max_batch_sizes = model_config_params.pop("max_batch_size", None)
                self._non_max_batch_size_param_combos = (
                    GeneratorUtils.generate_combinations(model_config_params)
                )
            else:
                if self._search_disabled:
                    self._non_max_batch_size_param_combos = (
                        self._generate_search_disabled_param_combos()
                    )
                else:
                    raise TritonModelAnalyzerException(
                        f"Automatic search not supported in ManualModelConfigGenerator"
                    )

    def _generate_search_disabled_param_combos(self) -> List[Dict]:
        """Return the configs when we want to search but searching is disabled"""
        return [DEFAULT_CONFIG_PARAMS]
