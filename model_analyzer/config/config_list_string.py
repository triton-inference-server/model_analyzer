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
    MODEL_ANALYZER_FAILURE


class ConfigListString(ConfigValue):
    """
    A list of string values.
    """

    def __init__(self,
                 preprocess=None,
                 required=False,
                 validator=None,
                 output_mapper=None):
        """
        Instantiate a new ConfigListString

        preprocess : callable
            Function be called before setting new values.
        required : bool
            Whether a given config is required or not.
        validator : callable or None
            A validator for the value of the field.
        output_mapper: callable or None
            This callable unifies the output value of this field.
        """

        # default validator
        if validator is None:
            def validator(x):
                return type(x) is list and len(x) > 0

        super().__init__(preprocess, required, validator, output_mapper)
        self._type = str
        self._value = []

    def set_value(self, value):
        """
        Set the value for this field.

        Parameters
        ----------
        value : object
            The value for this field. It can be a string of comma-delimited
            items or a list.

        Returns
        -------
        int
            1 on success, and 0 on failure
        """

        new_value = []
        if self._is_string(value):
            value = value.split(',')
            for item in value:
                new_value.append(self._type(item))
        elif self._is_list(value):
            for item in value:
                if not self._is_primitive(item):
                    return MODEL_ANALYZER_FAILURE
                new_value.append(self._type(item))
        else:
            if self._is_dict(value):
                return MODEL_ANALYZER_FAILURE
            new_value = [self._type(value)]

        return super().set_value(new_value)
