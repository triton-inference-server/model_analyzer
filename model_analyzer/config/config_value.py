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
    def __init__(self, preprocess=None, required=False):
        """
        Parameters
        ----------
        name : str
            Configuration name.
        preprocess : callable
            Function be called before setting new values.
        required : bool
            Whether a given config is required or not.
        """

        self._preprocess = preprocess
        self._required = required

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

        return_result = self._value
        if type(self._value) is dict:
            return_result = {}
            for key, value_ in self._value.items():
                if hasattr(value_, 'value'):
                    return_result[key] = value_.value()
                else:
                    return_result[key] = value_.value()
        elif type(self._value) is list:
            return_result = []
            for item in self._value:
                if hasattr(item, 'value'):
                    return_result.append(item.value())
                else:
                    return_result.append(item)

        return return_result

    def _is_primitive(self, value):
        """
        Is the value a primitive type.

        Parameters
        ----------
        value : object
            Value to be checked

        Returns
        -------
        bool
            True if yes, False if no
        """

        return not (self._is_dict(value) or self._is_list(value))

    def _is_string(self, value):
        """
        Is the value a string.

        Parameters
        ----------
        value : object
            Value to be checked

        Returns
        -------
        bool
            True if yes, False if no
        """

        return type(value) is str

    def _is_dict(self, value):
        """
        Is the value a dictionary.

        Parameters
        ----------
        value : object
            Value to be checked

        Returns
        -------
        bool
            True if yes, False if no
        """

        return type(value) is dict

    def _is_list(self, value):
        """
        Is the value a list.

        Parameters
        ----------
        value : object
            Value to be checked

        Returns
        -------
        bool
            True if yes, False if no
        """

        return type(value) is list

    def cli_type(self):
        """
        Get the corresponding CLI type for this field.

        Returns
        -------
        type
            Type to be used for the CLI.
        """

        return self._type

    def required(self):
        """
        Get the required field value

        Returns
        -------
        bool
            Whether the config field is required or not.
        """

        return self._required
