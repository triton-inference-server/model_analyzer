# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
from typing import List
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.objects.config_model_profile_spec import ConfigModelProfileSpec
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.device.gpu_device import GPUDevice


class ModelProfileSpec(ConfigModelProfileSpec):
    """
    The profile configuration and default model config for a single model to be profiled
    """

    def __init__(self, spec: ConfigModelProfileSpec,
                 config: ConfigCommandProfile, client: TritonClient,
                 gpus: List[GPUDevice]):
        self.__dict__ = deepcopy(spec.__dict__)

        self._default_model_config = ModelConfig.create_model_config_dict(
            config, client, gpus, config.model_repository, spec.model_name())

        if spec.model_name() in config.cpu_only_composing_models:
            self._cpu_only = True

    def get_default_config(self) -> dict:
        """ Returns the default configuration for this model """
        return deepcopy(self._default_model_config)

    def supports_batching(self) -> bool:
        """ Returns True if this model supports batching. Else False """
        if "max_batch_size" not in self._default_model_config or self._default_model_config[
                'max_batch_size'] == 0:
            return False
        return True

    def supports_dynamic_batching(self) -> bool:
        """ Returns True if this model supports dynamic batching. Else False """
        supports_dynamic_batching = self.supports_batching()

        if "sequence_batching" in self._default_model_config:
            supports_dynamic_batching = False
        return supports_dynamic_batching

    def is_ensemble(self) -> bool:
        """ Returns true if the model is an ensemble """
        return ("ensemble_scheduling" in self._default_model_config)
