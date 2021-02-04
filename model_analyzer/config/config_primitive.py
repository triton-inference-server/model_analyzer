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
from model_analyzer.constants import \
    MODEL_ANALYZER_SUCCESS, MODEL_ANALYZER_FAILURE


class ConfigPrimitive(ConfigValue):
    """
    A wrapper class for the primitive datatypes.
    """

    def __init__(
        self,
        type_,
        preprocess=None,
        required=False,
        validator=None,
    ):
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
        """

        # default validator
        if validator is None:
            def validator(x):
                return x is not None or x != ''

        super().__init__(preprocess, required, validator)

        self._type = type_
        self._value = None

    def set_value(self, value):
        """
        Set the value for this field.

        value : object
            The value for this field.
        """

        if self._is_primitive(value):
            value = self._type(value)
            return super().set_value(value)
        else:
            return MODEL_ANALYZER_FAILURE
