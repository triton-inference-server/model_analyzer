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

from model_analyzer.constants import \
    MODEL_ANALYZER_SUCCESS, MODEL_ANALYZER_FAILURE


class ConfigValue(abc.ABC):
    """
    Parent class for all the types used in the ConfigField.
    """

    def __init__(self,
                 preprocess=None,
                 required=False,
                 validator=None,
                 output_mapper=None):
        """
        Parameters
        ----------
        name : str
            Configuration name.
        preprocess : callable or None
            Function be called before setting new values.
        required : bool
            Whether a given config is required or not.
        validator : callable or None
            A validator for the value of the field.
        output_mapper: callable or None
            This callable unifies the output value of this field.
        """

        self._preprocess = preprocess
        self._required = required
        self._validator = validator
        self._output_mapper = output_mapper

    @abstractmethod
    def set_value(self, value):
        """
        Set the value for this field. This method must be implemented in each
        subclass.
        """

        output_validated = False
        if self._validator:
            if self._validator(value):
                output_validated = True
            else:
                return MODEL_ANALYZER_FAILURE
        else:
            output_validated = True

        self._value = value

        if output_validated:
            return MODEL_ANALYZER_SUCCESS
        else:
            return MODEL_ANALYZER_FAILURE

    def value(self):
        """
        Get the value of the config field.

        Returns
        -------
        object
            The value of the config field.
        """

        return_result = self._value
        if self._output_mapper:
            return_result = self._output_mapper(return_result)

        if type(return_result) is dict:
            final_return_result = {}
            for key, value_ in return_result.items():
                if hasattr(value_, 'value'):
                    final_return_result[key] = value_.value()
                else:
                    final_return_result[key] = value_.value()
            return_result = final_return_result
        elif type(return_result) is list:
            return_results = []
            for item in return_result:
                if hasattr(item, 'value'):
                    return_results.append(item.value())
                else:
                    return_results.append(item)
            return_result = return_results
        elif hasattr(return_result, 'value'):
            return_result = return_result.value()

        return return_result

    def raw_value(self):
        return self._value

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

        return self._cli_type

    def container_type(self):
        """
        Get the container type for this field.

        Returns
        -------
        ConfigValue
            Container type for the field.
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
