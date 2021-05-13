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

from .config_status import ConfigStatus
from .config_value import ConfigValue
from model_analyzer.constants import \
    CONFIG_PARSER_FAILURE, CONFIG_PARSER_SUCCESS


class ConfigUnion(ConfigValue):
    """
    ConfigUnion allows the value to be any of multiple ConfigValue types.
    """

    def __init__(self,
                 types,
                 preprocess=None,
                 required=False,
                 validator=None,
                 output_mapper=None,
                 name=None):
        """
        Create a new ConfigUnion.

        Parameters
        ----------
        types : list
            A list of ConfigValue that are allowed for this field.
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

        self._types = types
        self._used_type_index = 0
        super().__init__(preprocess, required, validator, output_mapper, name)

    def set_value(self, value):
        """
        Set the value for this field.

        value : object
            The value for this field.
        """

        config_statuses = []
        for i, type_ in enumerate(self._types):
            config_status = type_.set_value(value)
            config_statuses.append(config_status)
            if config_status.status() == CONFIG_PARSER_SUCCESS:
                self._used_type_index = i
                return super().set_value(type_)
        else:
            message = (
                f'Value "{value}" cannot be set for field "{self.name()}".'
                ' This field allows multiple types of values.'
                ' You only need to fix one of the errors below:\n')
            for config_status in config_statuses:
                message_lines = config_status.message().split('\n')

                # ConfigUnion needs to repeat the same structure. The lines
                # below make a couple of adjustments to ensure that
                # lines are printed correctly.
                if type(config_status.config_object()) is ConfigUnion:

                    # Make sure that the line is not empty
                    if not message_lines[0].strip() == '':
                        message += f"\t* {message_lines[0]}\n"
                        for message_line in message_lines[1:]:
                            message += f"\t {message_line}\n"
                else:
                    for message_line in message_lines:
                        message += f"\t* {message_line} \n"

            return ConfigStatus(CONFIG_PARSER_FAILURE, message, self)

    def set_name(self, name):
        """
        This function must be called before the set_value.
        """

        super().set_name(name)
        for type_ in self._types:
            type_.set_name(self.name())

    def cli_type(self):
        used_type_index = self._used_type_index
        return self._types[used_type_index].cli_type()

    def container_type(self):
        return self.cli_type()
