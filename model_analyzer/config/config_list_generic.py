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

from .config_value import ConfigValue
from model_analyzer.constants import \
     MODEL_ANALYZER_SUCCESS, MODEL_ANALYZER_FAILURE

from copy import deepcopy


class ConfigListGeneric(ConfigValue):
    """
    A generic list.
    """
    def __init__(self, types, preprocess=None, required=False):
        """
        Create a new list of numeric values.

        Parameters
        ----------
        types : A list of allowed types
            The type of elements in the list
        preprocess : callable
            Function be called before setting new values.
        required : bool
            Whether a given config is required or not.
        """

        super().__init__(preprocess, required)

        self._type = str
        self._allowed_types = types
        self._value = []

    def set_value(self, value):
        """
        Set the value for this field.

        Parameters
        ----------
        value : object
            The value for this field.
        """

        allowed_types = self._allowed_types

        new_value = []
        if type(value) is list:
            for item in value:
                # Try setting the value for each type in the list, if none
                # of them are possible we fail setting the value for this field
                for allowed_type in allowed_types:
                    list_item = deepcopy(allowed_type)
                    status = list_item.set_value(item)
                    if status == MODEL_ANALYZER_SUCCESS:
                        new_value.append(list_item)
                        break
                else:
                    return MODEL_ANALYZER_FAILURE
        else:
            return MODEL_ANALYZER_FAILURE

        self._value = new_value
        return MODEL_ANALYZER_SUCCESS
