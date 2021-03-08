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
from .config_status import ConfigStatus
from model_analyzer.constants import \
    CONFIG_PARSER_FAILURE, CONFIG_PARSER_SUCCESS


class ConfigListNumeric(ConfigValue):
    """
    A list of numeric values.
    """

    def __init__(self,
                 type_,
                 preprocess=None,
                 required=False,
                 validator=None,
                 output_mapper=None,
                 name=None):
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
        output_mapper: callable or None
            This callable unifies the output value of this field.
        name : str
            Fully qualified name for this field.
        """

        # default validator
        if validator is None:

            def validator(x):
                if type(x) is list:
                    return ConfigStatus(CONFIG_PARSER_SUCCESS)

                return ConfigStatus(
                    CONFIG_PARSER_FAILURE,
                    f'The value for field "{self.name()}" should be a list'
                    ' and the length must be larger than zero.')

        super().__init__(preprocess, required, validator, output_mapper, name)
        self._type = type_
        self._cli_type = str
        self._value = []

    def _process_list(self, value):
        """
        A function to process the case where value is
        a list.
        """

        type_ = self._type
        new_value = []

        for item in value:
            item = type_(item)
            new_value.append(item)

        return new_value

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

        try:
            if self._is_string(value):
                self._value = []
                value = value.split(',')

            if self._is_list(value):
                new_value = self._process_list(value)

            elif self._is_dict(value):
                two_key_condition = len(
                    value) == 2 and 'start' in value and 'stop' in value
                three_key_condition = len(
                    value) == 3 and 'start' in value and 'stop' in value\
                    and 'step' in value
                if two_key_condition or three_key_condition:
                    step = 1
                    start = int(value['start'])
                    stop = int(value['stop'])
                    if start > stop:
                        return ConfigStatus(
                            CONFIG_PARSER_FAILURE,
                            f'When a dictionary is used for field "{self.name()}",'
                            ' "start" should be less than "stop".'
                            f' Current value is {value}.',
                            config_object=self)

                    if 'step' in value:
                        step = int(value['step'])
                    new_value = list(range(start, stop + 1, step))
                else:
                    return ConfigStatus(
                        CONFIG_PARSER_FAILURE,
                        f'If a dictionary is used for field "{self.name()}", it'
                        ' should only contain "start" and "stop" key with an'
                        f' optional "step" key. Currently, contains {list(value)}.',
                        config_object=self)
            else:
                new_value = [type_(value)]
        except ValueError as e:
            message = f'Failed to set the value for field "{self.name()}". Error: {e}.'
            return ConfigStatus(CONFIG_PARSER_FAILURE, message, self)

        return super().set_value(new_value)
