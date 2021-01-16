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

import abc
from abc import abstractmethod


class ConfigValue(abc.ABC):
    """
    Parent class for all the types used in the ConfigField.
    """
    @abstractmethod
    def set_value(self, value):
        """
        Set the value for this field. This method must be implemented in each
        subclass.
        """

        pass

    def value(self):
        """
        Get the value of the config field.

        Returns
        -------
        object
            The value of the config field.
        """

        return self._value

    def cli_type(self):
        """
        Get the corresponding CLI type for this field.

        Returns
        -------
        type
            Type to be used for the CLI.
        """

        return self._type
