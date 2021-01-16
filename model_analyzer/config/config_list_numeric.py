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


class ConfigListNumeric(ConfigValue):
    """
    A list of numeric values.
    """
    def __init__(self, type_):
        """
        Create a new list of numeric values.

        Parameters
        ----------
        type_ : type
            The type of elements in the list
        """

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

        # Is a comma delimited list?
        if type(value) is str:
            self._value = []
            value = value.split(',')
            for item in value:
                self._value.append(type_(item))
        # Is a list of values?
        elif type(value) is list:
            self._value = []
            for item in value:
                self._value.append(type_(item))
        # Is a range?
        elif type(value) is dict:
            if 'start' in value and 'end' in value:
                step = 1
                start = value['start']
                end = value['end']
                if 'step' in value:
                    step = value['step']
                self._value = list(range(start, end, step))
        else:
            self._value = type_(value)

    def cli_type(self):
        """
        Get the type of this field for CLI.

        Returns
        -------
        type
            str
        """

        return str
