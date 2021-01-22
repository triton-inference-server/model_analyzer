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

from model_analyzer.model_analyzer_exceptions import \
    TritonModelAnalyzerException
from model_analyzer.constants import \
     MODEL_ANALYZER_SUCCESS


class ConfigField:
    def __init__(self,
                 name=None,
                 flags=None,
                 choices=None,
                 description=None,
                 default_value=None,
                 field_types=None,
                 parser_args=None):
        """
        Create a configuration field.

        Parameters
        ----------
        name : str
            Configuration name.
        description : str
            Description of the config.
        flags : list
            List of the flags to be used for the CLI.
        field_types : list
            List of type ConfigValue.
        choices : list or None
            List of the choices to be used.
        preprocess : callable
            Function be called before setting new values.
        default_value : object
            Default value used for the config field.
        parser_args : dict
            Additionaly arguments to be passed to ArgumentParser.
        """

        self._description = description
        self._default_value = default_value
        self._name = name
        self._field_types = field_types
        self._used_field_idx = 0
        self._flags = flags
        self._choices = choices
        self._parser_args = {} if parser_args is None else parser_args

    def choices(self):
        """
        List of choices that are allowed
        """

        return self._choices

    def parser_args(self):
        """
        Get the additional arguments to be passed to ArgumentParser.

        Returns
        -------
        dict or None
            A dictionary where the keys are arguments and the values are
            the argument values
        """

        return self._parser_args

    def description(self):
        """
        Get the field description.

        Returns
        -------
        str or None
            Description of the config field
        """

        return self._description

    def field_type(self):
        """
        Get the field type.

        Returns
        -------
        ConfigValue
            Type of the config field
        """

        used_field_idx = self._used_field_idx
        return self._field_types[used_field_idx].cli_type()

    def name(self):
        """
        Get the name of the config field.

        Returns
        -------
        str
            Name of the config field
        """

        return self._name

    def default_value(self):
        """
        Get the value for this config field.

        Returns
        -------
        object
            The default value of the config field
        """

        return self._default_value

    def flags(self):
        """
        Get the CLI flags.

        Returns
        -------
        list
            A list of the CLI flags used
        """

        return self._flags

    def set_value(self, value):
        """
        Set the value for the config field.
        """

        # Trying setting the value for each type in the field. If it
        # was not succesful for any of the types, raise an exception.
        for i, config_type in enumerate(self._field_types):
            status = config_type.set_value(value)
            if status == MODEL_ANALYZER_SUCCESS:
                self._used_field_idx = i
                break
        else:
            raise TritonModelAnalyzerException(
                f'Can\'t set value {{ {value} }} for field {self._name}')

    def value(self):
        """
        Get the value for the config field.

        Returns
        -------
        object
            The value of the config field.
        """

        used_field_idx = self._used_field_idx
        return self._field_types[used_field_idx].value()

    def required(self):
        """
        Get the required field value

        Returns
        -------
        bool
            Whether the config field is required or not.
        """

        used_field_idx = self._used_field_idx
        return self._field_types[used_field_idx].required()
