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
from model_analyzer.constants import MODEL_ANALYZER_FAILURE


class ConfigEnum(ConfigValue):
    """
    Enum type support for config.
    """

    def __init__(self,
                 choices,
                 preprocess=None,
                 required=False,
                 validator=None,
                 output_mapper=None):
        """
        Create a new enum config field.

        Parameters
        ----------
        types : A list of allowed types
            The type of elements in the list
        preprocess : callable
            Function be called before setting new values.
        required : bool
            Whether a given config is required or not.
        validator : callable or None
            A validator for the final value of the field.
        output_mapper: callable
            This callable unifies the output value of this field.
        """

        self._choices = choices
        self._type = self
        super().__init__(preprocess, required, validator, output_mapper)

    def set_value(self, value):
        choices = self._choices

        if value not in choices:
            return MODEL_ANALYZER_FAILURE

        return super().set_value(value)

    def cli_type(self):
        """
        Get the type of this field for CLI.

        Returns
        -------
        type
            str
        """

        return str
