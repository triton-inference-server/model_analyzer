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
    MODEL_ANALYZER_FAILURE
from copy import deepcopy


class ConfigObject(ConfigValue):
    """
    Representation of dictionaries in Config
    """

    def __init__(self,
                 schema,
                 preprocess=None,
                 required=False,
                 validator=None,
                 output_mapper=None
                 ):
        """
        schema : dict
            A dictionary where the keys are the object keys and the values
            are of type ConfigValue
        preprocess : callable
            Function be called before setting new values.
        required : bool
            Whether a given config is required or not.
        validator : callable or None
            A validator for the value of the field.
        output_mapper: callable or None
            This callable unifies the output value of this field.
        """

        # default validator
        if validator is None:
            def validator(x):
                return type(x) is dict and len(x) > 0

        super().__init__(preprocess, required, validator, output_mapper)
        self._type = self
        self._cli_type = str
        self._value = {}
        self._schema = schema

    def set_value(self, value):
        """
        Set the value for this field.

        Parameters
        ----------
        value : object
            The value for this field.

        Returns
        -------
        int
            1 on success, and 0 on failure
        """

        new_value = {}
        schema = self._schema

        if type(value) is dict:
            for key, value_ in value.items():
                if key in schema:
                    new_item = deepcopy(schema[key])

                # If the key is not available in the schema, but wildcard is
                # present, we use the schema for the wildcard.
                elif '*' in schema:
                    new_item = deepcopy(schema['*'])
                else:
                    return MODEL_ANALYZER_FAILURE

                new_value[key] = new_item

                # Set the value of the new field
                status = new_item.set_value(value_)

                # If it was not able to set the value, for this
                # field, we fail.
                if status == MODEL_ANALYZER_FAILURE:
                    return MODEL_ANALYZER_FAILURE
        else:
            return MODEL_ANALYZER_FAILURE
        return super().set_value(new_value)

    def __getattr__(self, name):
        # We need to add this check, to prevent infinite
        # recursion when using deep copy. See the link below:
        # https://stackoverflow.com/questions/47299243/recursionerror-when-python-copy-deepcopy

        if name == "__setstate__":
            raise AttributeError(name)
        elif name in self._value:
            return self._value[name].value()
        else:
            raise AttributeError(name)
