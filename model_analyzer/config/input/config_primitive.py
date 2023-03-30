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
from model_analyzer.constants import CONFIG_PARSER_FAILURE


class ConfigPrimitive(ConfigValue):
    """
    A wrapper class for the primitive datatypes.
    """

    def __init__(self,
                 type_,
                 preprocess=None,
                 required=False,
                 validator=None,
                 output_mapper=None,
                 name=None):
        """
        Parameters
        ----------
        type_ : type
            Type of the field.
        preprocess : callable
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

        super().__init__(preprocess, required, validator, output_mapper, name)

        self._type = self._cli_type = type_
        self._value = None

    def set_value(self, value):
        """
        Set the value for this field.

        value : object
            The value for this field.
        """

        if self._is_primitive(value):
            try:
                value = self._type(value)
            except ValueError as e:
                message = f'Failed to set the value for field "{self.name()}". Error: {e}.'
                return ConfigStatus(CONFIG_PARSER_FAILURE, message, self)
            return super().set_value(value)
        else:
            return ConfigStatus(
                CONFIG_PARSER_FAILURE,
                f'Value "{value}" for field "{self.name()}" should be a primitive type.',
                self)
