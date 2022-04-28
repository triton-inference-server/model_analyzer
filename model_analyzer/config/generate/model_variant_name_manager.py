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

import collections.abc
from copy import deepcopy
from model_analyzer.constants import DEFAULT_CONFIG_PARAMS


class ModelVariantNameManager:

    def __init__(self):

        # Stored as _variant_names[base_model_name][param_combo] = variant_name_string
        self._variant_names = {}

        # Stored as _model_name_index[base_model_name] = current_count_integer
        self._model_name_index = {}

    def get_model_variant_name(self, base_model_name, param_combo):
        """
        Given a base model name and a dict of parameters to be applied
        to the base model config, return the name of the model variant

        If the same input values are provided to this function multiple times, 
        the same value will be returned
        """
        if self._model_variant_exists(base_model_name, param_combo):
            return self._get_model_variant_name(base_model_name, param_combo)
        else:
            return self._make_model_variant_name(base_model_name, param_combo)

    def _model_variant_exists(self, base_model_name, param_combo):
        """ 
        Returns true if a model variant name already exists for this combination 
        """
        hashable_key = self._get_hashable_key(param_combo)

        return base_model_name in self._variant_names \
           and hashable_key in self._variant_names[base_model_name]

    def _get_model_variant_name(self, base_model_name, param_combo):
        """
        Returns the model variant name that already exists for this combination
        """
        hashable_key = self._get_hashable_key(param_combo)
        return self._variant_names[base_model_name][hashable_key]

    def _make_model_variant_name(self, base_model_name, param_combo):
        """
        Create and save a new model variant for this base model
        """
        if param_combo == DEFAULT_CONFIG_PARAMS:
            variant_name = f'{base_model_name}_config_default'
        else:
            model_name_index = self._get_next_model_name_index(base_model_name)
            variant_name = f'{base_model_name}_config_{model_name_index}'
            self._add_model_variant_name(base_model_name, param_combo,
                                         variant_name)
        return variant_name

    def _add_model_variant_name(self, base_model_name, param_combo,
                                variant_name):
        """ 
        Save a new model variant name for this base model
        """
        if base_model_name not in self._variant_names:
            self._variant_names[base_model_name] = {}

        hashable_key = self._get_hashable_key(param_combo)
        self._variant_names[base_model_name][hashable_key] = variant_name

    def _get_next_model_name_index(self, base_model_name):
        """
        Get the next index to be used as a unique number for variant naming
        """
        if base_model_name not in self._model_name_index:
            self._model_name_index[base_model_name] = 0
        else:
            self._model_name_index[base_model_name] += 1

        return self._model_name_index[base_model_name]

    def _get_hashable_key(self, obj):
        obj_copy = deepcopy(obj)

        return self._get_hashable_key_helper(obj_copy)

    def _get_hashable_key_helper(self, obj):
        """
        Given a list or dict which may have nested dicts/lists, return a key that 
        is hashable 
        """

        if isinstance(obj, collections.abc.Hashable):
            key = obj
        elif isinstance(obj, collections.abc.Mapping):
            key = frozenset(
                (k, self._get_hashable_key_helper(v)) for k, v in obj.items())
        elif isinstance(obj, collections.abc.Iterable):
            key = tuple(self._get_hashable_key_helper(item) for item in obj)
        else:
            raise TypeError(type(obj))

        return key
