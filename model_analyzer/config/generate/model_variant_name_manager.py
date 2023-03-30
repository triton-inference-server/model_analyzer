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

from typing import Dict, Tuple, List
from copy import deepcopy
from model_analyzer.constants import DEFAULT_CONFIG_PARAMS


class ModelVariantNameManager:

    def __init__(self) -> None:

        # Dict of {model_config_name: model_config_dict}
        self._model_config_dicts: Dict[str, Dict] = {}

        # Dict of {base_model_name: current_count_integer}
        self._model_name_index: Dict[str, int] = {}

    @classmethod
    def from_dict(
            cls,
            model_variant_name_manager_dict: Dict) -> "ModelVariantNameManager":
        model_variant_name_manager = ModelVariantNameManager()

        model_variant_name_manager._model_config_dicts = model_variant_name_manager_dict[
            '_model_config_dicts']
        model_variant_name_manager._model_name_index = model_variant_name_manager_dict[
            '_model_name_index']

        return model_variant_name_manager

    @staticmethod
    def make_ensemble_composing_model_key(
            ensemble_config_dicts: List[Dict]) -> Dict[str, str]:
        ensemble_names = [
            ensemble_config_dict["name"]
            for ensemble_config_dict in ensemble_config_dicts
        ]
        ensemble_key = ','.join(ensemble_names)

        return {"key": ensemble_key}

    def get_model_variant_name(self, model_name: str, model_config_dict: Dict,
                               param_combo: Dict) -> Tuple[bool, str]:
        """
        Given a base model name and a dict of parameters to be applied
        to the base model config, return if the variant already existed 
        and the name of the model variant

        If the same input values are provided to this function multiple times, 
        the same value will be returned
        """
        return self._get_variant_name(model_name,
                                      model_config_dict,
                                      is_ensemble=False,
                                      param_combo=param_combo)

    def get_ensemble_model_variant_name(
            self, model_name: str, ensemble_dict: Dict) -> Tuple[bool, str]:
        """
        Given a base ensemble model name and a dict of ensemble composing configs,
        return if the variant already existed and the name of the model variant

        If the same input values are provided to this function multiple times,
        the same value will be returned
        """
        return self._get_variant_name(model_name,
                                      ensemble_dict,
                                      is_ensemble=True)

    def _get_variant_name(self,
                          model_name: str,
                          config_dict: Dict,
                          is_ensemble: bool,
                          param_combo: Dict = {}) -> Tuple[bool, str]:
        model_config_dict = self._copy_and_restore_model_config_dict_name(
            model_name, config_dict)

        variant_found, model_variant_name = self._find_existing_variant(
            model_config_dict)

        if is_ensemble:
            if self._is_ensemble_default_config(config_dict):
                return (False, model_name + '_config_default')
        else:
            if self._is_default_config(param_combo):
                return (False, model_name + '_config_default')

        if variant_found:
            return (True, model_variant_name)

        model_variant_name = self._create_new_model_variant(
            model_name, model_config_dict)

        return (False, model_variant_name)

    def _copy_and_restore_model_config_dict_name(
            self, model_name: str, model_config_dict: Dict) -> Dict:
        model_config_dict_copy = deepcopy(model_config_dict)
        model_config_dict_copy['name'] = model_name

        return model_config_dict_copy

    def _find_existing_variant(self,
                               model_config_dict: Dict) -> Tuple[bool, str]:
        for model_config_name, model_config_variant_dict in self._model_config_dicts.items(
        ):
            if model_config_dict == model_config_variant_dict:
                return (True, model_config_name)

        return (False, "")

    def _is_default_config(self, param_combo: Dict) -> bool:
        return param_combo == DEFAULT_CONFIG_PARAMS

    def _is_ensemble_default_config(self, ensemble_dict: Dict) -> bool:
        return '_config_default' in ensemble_dict['key']

    def _create_new_model_variant(self, model_name: str,
                                  model_config_dict: Dict) -> str:
        if model_name not in self._model_name_index:
            new_index = 0
        else:
            new_index = self._model_name_index[model_name] + 1

        self._model_name_index[model_name] = new_index
        model_config_name = model_name + '_config_' + str(new_index)
        self._model_config_dicts[model_config_name] = model_config_dict

        return model_config_name
