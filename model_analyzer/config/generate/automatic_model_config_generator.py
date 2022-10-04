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

from typing import List, Dict, Any
from .base_model_config_generator import BaseModelConfigGenerator
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.config.generate.model_variant_name_manager import ModelVariantNameManager
from model_analyzer.constants import LOGGER_NAME, DEFAULT_CONFIG_PARAMS
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.triton.model.model_config import ModelConfig
from .model_profile_spec import ModelProfileSpec
import logging

logger = logging.getLogger(LOGGER_NAME)


class AutomaticModelConfigGenerator(BaseModelConfigGenerator):
    """ Given a model, generates model configs in automatic search mode """

    def __init__(self, config: ConfigCommandProfile, gpus: List[GPUDevice],
                 model: ModelProfileSpec, client: TritonClient,
                 model_variant_name_manager: ModelVariantNameManager,
                 default_only: bool, early_exit_enable: bool) -> None:
        """
        Parameters
        ----------
        config: ModelAnalyzerConfig
        gpus: List of GPUDevices
        model: ModelProfileSpec
            The model to generate ModelConfigs for
        client: TritonClient
        model_variant_name_manager: ModelVariantNameManager
        default_only: Bool
            If true, only the default config will be generated
            If false, the default config will NOT be generated
        early_exit_enable: Bool
            If true, the generator can early exit if throughput plateaus
        """
        super().__init__(config, gpus, model, client,
                         model_variant_name_manager, default_only,
                         early_exit_enable)

        self._max_instance_count = config.run_config_search_max_instance_count
        self._min_instance_count = config.run_config_search_min_instance_count
        self._max_model_batch_size = config.run_config_search_max_model_batch_size
        self._min_model_batch_size = config.run_config_search_min_model_batch_size

        self._instance_kind = "KIND_CPU" if self._cpu_only else "KIND_GPU"

        self._curr_instance_count = self._min_instance_count
        self._curr_max_batch_size = 0

        self._reset_max_batch_size()

        if not self._early_exit_enable:
            raise TritonModelAnalyzerException(
                "Early exit disable is not supported in automatic model config generator"
            )

    def _done_walking(self) -> bool:
        return self._curr_instance_count > self._max_instance_count

    def _step(self) -> None:
        self._step_max_batch_size()

        if self._done_walking_max_batch_size():
            self._reset_max_batch_size()
            self._step_instance_count()

    def _step_max_batch_size(self) -> None:
        self._curr_max_batch_size *= 2

        last_max_throughput = self._get_last_results_max_throughput()
        if last_max_throughput:
            self._curr_max_batch_size_throughputs.append(last_max_throughput)

    def _step_instance_count(self) -> None:
        self._curr_instance_count += 1

    def _done_walking_max_batch_size(self) -> bool:
        if self._last_results_erroneous():
            return True

        if self._max_batch_size_limit_reached():
            return True

        if not self._last_results_increased_throughput():
            self._print_max_batch_size_plateau_warning()
            return True

        return False

    def _max_batch_size_limit_reached(self) -> bool:
        return self._curr_max_batch_size > self._max_model_batch_size

    def _reset_max_batch_size(self) -> None:
        super()._reset_max_batch_size()

        if self._base_model.supports_batching():
            self._curr_max_batch_size = self._min_model_batch_size
        else:
            self._curr_max_batch_size = self._max_model_batch_size

    def _get_next_model_config(self) -> ModelConfig:
        param_combo = self._get_curr_param_combo()
        model_config = self._make_direct_mode_model_config(param_combo)
        return model_config

    def _get_curr_param_combo(self) -> Dict:
        if self._default_only:
            return DEFAULT_CONFIG_PARAMS

        config: Dict[str, Any] = {
            'instance_group': [{
                'count': self._curr_instance_count,
                'kind': self._instance_kind
            }]
        }

        if self._base_model.supports_batching():
            config['max_batch_size'] = self._curr_max_batch_size

        if self._base_model.supports_dynamic_batching():
            config['dynamic_batching'] = {}

        return config
