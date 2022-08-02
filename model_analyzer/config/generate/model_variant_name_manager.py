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

from typing import Dict
import collections.abc
from copy import deepcopy
from model_analyzer.constants import DEFAULT_CONFIG_PARAMS


class ModelVariantNameManager:

    def __init__(self):

        # Dict of {model_config_name: model_config_dict}
        self._model_config_dicts = {}

        # Dict of {base_model_name: current_count_integer}
        self._model_name_index = {}

    @classmethod
    def _from_dict(
            cls,
            model_variant_name_manager_dict: Dict) -> "ModelVariantNameManager":
        model_variant_name_manager = ModelVariantNameManager()

        model_variant_name_manager._model_config_dicts = model_variant_name_manager_dict[
            '_model_config_dicts']
        model_variant_name_manager._model_name_index = model_variant_name_manager_dict[
            '_model_name_index']

        return model_variant_name_manager

    def get_model_variant_name(self, model_name: str, model_config_dict: Dict,
                               param_combo: Dict) -> str:
        """
        Given a base model name and a dict of parameters to be applied
        to the base model config, return the name of the model variant

        If the same input values are provided to this function multiple times, 
        the same value will be returned
        """

        new_mcd = deepcopy(model_config_dict)
        new_mcd['name'] = model_name

        # Find existing variant
        for model_config_name, model_config_variant_dict in self._model_config_dicts.items(
        ):
            if new_mcd == model_config_variant_dict:
                return model_config_name

        # Add new variant to list
        if model_name in self._model_name_index:
            if self._model_name_index[model_name] == 'default':
                self._model_name_index[model_name] = 0

                model_config_name = model_name + '_config_0'
                self._model_config_dicts[model_config_name] = new_mcd
            else:
                new_index = self._model_name_index[model_name] + 1
                self._model_name_index[model_name] = new_index

                model_config_name = model_name + '_config_' + str(new_index)
                self._model_config_dicts[model_config_name] = new_mcd

        else:
            model_config_name = model_name + '_config_default'

            self._model_config_dicts[model_config_name] = new_mcd

            self._model_name_index[model_name] = 'default'

        return model_config_name