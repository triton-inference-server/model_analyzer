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


class ConfigListString(ConfigValue):
    """
    A list of string values.
    """
    def __init__(self):
        """
        Instantiate a new ConfigListString
        """

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
        """

        # If the value is string, it should be a comma delimited list of values
        if type(value) is str:
            self._value = value.split(',')
        elif type(value) is list:
            self._value = value
        else:
            self._value = str(value)
