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

from model_analyzer.constants import \
    MODEL_ANALYZER_FAILURE, MODEL_ANALYZER_SUCCESS
from .config_value import ConfigValue


class ConfigUnion(ConfigValue):
    """
    ConfigUnion allows the value to be any of multiple ConfigValue types.
    """

    def __init__(self,
                 types,
                 preprocess=None,
                 required=False,
                 validator=None,
                 output_mapper=None
                 ):
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
        """

        self._types = types
        self._used_type_index = 0
        super().__init__(preprocess, required, validator, output_mapper)

    def set_value(self, value):
        """
        Set the value for this field.

        value : object
            The value for this field.
        """

        for i, type_ in enumerate(self._types):
            status = type_.set_value(value)
            if status == MODEL_ANALYZER_SUCCESS:
                self._used_type_index = i
                return super().set_value(type_)
        else:
            return MODEL_ANALYZER_FAILURE

    def cli_type(self):
        used_type_index = self._used_type_index
        return self._types[used_type_index].cli_type()

    def container_type(self):
        return self.cli_type()
