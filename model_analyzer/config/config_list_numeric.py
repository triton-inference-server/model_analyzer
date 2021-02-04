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


class ConfigListNumeric(ConfigValue):
    """
    A list of numeric values.
    """

    def __init__(self, type_, preprocess=None, required=False, validator=None):
        """
        Create a new list of numeric values.

        Parameters
        ----------
        type_ : type
            The type of elements in the list
        preprocess : callable
            Function be called before setting new values.
        required : bool
            Whether a given config is required or not.
        validator : callable or None
            A validator for the final value of the field.
        """

        # default validator
        if validator is None:

            def validator(x):
                if type(x) is list and len(x) > 0:
                    return True

                return False

        super().__init__(preprocess, required, validator)
        self._type = type_
        self._value = []

    def set_value(self, value):
        """
        Set the value for this field.

        Parameters
        ----------
        value : object
            The value for this field. It can be comma delimited list, or an
            array, or a range
        """

        type_ = self._type
        new_value = []

        if self._is_string(value):
            self._value = []
            value = value.split(',')
            for item in value:
                new_value.append(type_(item))
        elif self._is_list(value):
            for item in value:
                new_value.append(type_(item))
        elif self._is_dict(value):
            two_key_condition = len(
                value) == 2 and 'start' in value and 'stop' in value
            three_key_condition = len(
                value) == 3 and 'start' in value and 'stop' in value\
                and 'step' in value
            if two_key_condition or three_key_condition:
                step = 1
                start = value['start']
                stop = value['stop']
                if 'step' in value:
                    step = value['step']
                new_value = list(range(start, stop + 1, step))
            else:
                return MODEL_ANALYZER_FAILURE
        else:
            new_value = [type_(value)]

        return super().set_value(new_value)

    def cli_type(self):
        """
        Get the type of this field for CLI.

        Returns
        -------
        type
            str
        """

        return str
