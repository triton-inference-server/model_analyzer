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
    CONFIG_PARSER_SUCCESS, CONFIG_PARSER_FAILURE
from .config_list_generic import ConfigListGeneric
from .config_status import ConfigStatus


class ConfigSweep(ConfigValue):
    """
    Representation of dictionaries in Config
    """

    def __init__(self,
                 sweep_type,
                 preprocess=None,
                 required=False,
                 validator=None,
                 output_mapper=None):
        """
        sweep_type : ConfigValue
            The type of parameter that we are going to sweep on.
        preprocess : callable
            Function be called before setting new values.
        required : bool
            Whether a given config is required or not.
        validator : callable or None
            A validator for the value of the field.
        output_mapper: callable or None
            This callable unifies the output value of this field.
        """

        self._sweep_type = sweep_type
        self._is_sweepable = False
        super().__init__(preprocess, required, validator, output_mapper)

    def set_value(self, value):
        config_statuses = []

        sweep_type = self._sweep_type
        sweep_type.set_name(self._name)
        config_status = sweep_type.set_value(value)
        config_statuses.append(config_status)

        if config_status.status() == CONFIG_PARSER_SUCCESS:
            self._is_sweepable = False
            return super().set_value([sweep_type])
        else:
            config_list = ConfigListGeneric(sweep_type)
            config_list.set_name(self._name)
            config_status_list = config_list.set_value(value)
            if config_status_list.status() == CONFIG_PARSER_SUCCESS:
                self._is_sweepable = True
                return super().set_value(config_list)
            config_statuses.append(config_status_list)

        message = (
            f'Field "{self._name}" is a sweep parameter. If you intend to provide a sweep parameter, '
            'fix the number one error, otherwise fix error number two: '
            f'1. {config_statuses[0].message()}'
            f' 2. {config_statuses[1].message()}')

        return ConfigStatus(CONFIG_PARSER_FAILURE, message)

    def is_sweepable(self):
        return self._is_sweepable
