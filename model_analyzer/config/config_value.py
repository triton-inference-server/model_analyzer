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
    CONFIG_PARSER_FAILURE, CONFIG_PARSER_SUCCESS
from .config_status import ConfigStatus
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class ConfigValue(abc.ABC):
    """
    Parent class for all the types used in the ConfigField.
    """

    def __init__(self,
                 preprocess=None,
                 required=False,
                 validator=None,
                 output_mapper=None,
                 name=None):
        """
        Parameters
        ----------
        preprocess : callable or None
            Function be called before setting new values.
        required : bool
            Whether a given config is required or not.
        validator : callable or None
            A validator for the value of the field.
        output_mapper: callable or None
            This callable unifies the output value of this field.
        name : str
            Fully qualified name for this field.
        """

        self._preprocess = preprocess
        self._is_sweepable = False
        self._required = required
        self._validator = validator
        self._output_mapper = output_mapper
        self._name = name

    @abstractmethod
    def set_value(self, value):
        """
        Set the value for this field. This method must be implemented in each
        subclass.
        """

        if self._validator:
            config_status = self._validator(value)
            if config_status.status() == CONFIG_PARSER_FAILURE:
                return config_status

        self._value = value

        return ConfigStatus(status=CONFIG_PARSER_SUCCESS)

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
                    raise TritonModelAnalyzerException(
                        'ConfigObject should always have a '
                        '"value" attribute for each value.')
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
        Get the 'required' field value

        Returns
        -------
        bool
            Whether the config field is required or not.
        """

        return self._required

    def name(self):
        """
        Get the fully qualified name for this field.

        Returns
        -------
        str or None
            Fully qualified name for this field.
        """

        return self._name

    def set_name(self, name):
        """
        Set the name for this field.

        Parameters
        ----------
        name : str
            New name to be set for this field.
        """

        self._name = name
