# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from copy import deepcopy
from model_analyzer.config.input.objects.config_model_profile_spec import ConfigModelProfileSpec
from model_analyzer.triton.model.model_config import ModelConfig


class ModelProfileSpec(ConfigModelProfileSpec):
    # FIXME docs
    def __init__(self, spec, config, client, gpus):

        # FIXME proper way to do this
        self._model_name = spec._model_name
        self._cpu_only = spec._cpu_only
        self._objectives = spec._objectives
        self._constraints = spec._constraints
        self._parameters = spec._parameters
        self._model_config_parameters = spec._model_config_parameters
        self._perf_analyzer_flags = spec._perf_analyzer_flags
        self._triton_server_flags = spec._triton_server_flags
        self._triton_server_environment = spec._triton_server_environment

        self._default_model_config = ModelConfig.create_model_config_dict(
            config, client, gpus, config.model_repository, spec.model_name())

    def get_default_config(self) -> dict:
        return deepcopy(self._default_model_config)

    def supports_batching(self) -> bool:
        if "max_batch_size" not in self._default_model_config or self._default_model_config[
                'max_batch_size'] == 0:
            return False
        return True

    def supports_dynamic_batching(self) -> bool:
        supports_dynamic_batching = self.supports_batching()

        if "sequence_batching" in self._default_model_config:
            supports_dynamic_batching = False
        return supports_dynamic_batching
