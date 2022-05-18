# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

logger = logging.getLogger(LOGGER_NAME)


class YamlConfigValidator:
    """
    Validate all options from the yaml file.
    """

    _valid_yaml_options = set({})
    _initialized = False

    @staticmethod
    def is_yaml_file_valid(yaml_file):
        if not YamlConfigValidator._initialized:
            YamlConfigValidator._create_valid_option_set()

        valid = True
        for option in yaml_file.keys():
            valid &= YamlConfigValidator.is_valid_option(option)

        if not valid:
            raise TritonModelAnalyzerException(
                "The Yaml configuration file is incorrect. Please check the logged errors for the specific options that are causing the issue."
            )
        else:
            return True

    @staticmethod
    def is_valid_option(option):
        if not YamlConfigValidator._initialized:
            YamlConfigValidator._create_valid_option_set()

        if option not in YamlConfigValidator._valid_yaml_options:
            logger.error(
                f'{option} is not a valid yaml argument. Please be sure to use underscores and check spelling.'
            )
            return False
        return True

    @staticmethod
    def _create_valid_option_set():
        """
        Create set of valid yaml options
        """
        # Importing here to remove a circular dependency.
        # If these imports are moved to the top of the file, they are imported at file load.
        # This will cause a circular dependency between ConfigCommand, ConfigCommand*, and YamlConfigValidator
        # However, importing here, only requires these files to be imported on object initialization.
        from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
        from model_analyzer.config.input.config_command_analyze import ConfigCommandAnalyze
        from model_analyzer.config.input.config_command_report import ConfigCommandReport
        config_array = [
            ConfigCommandProfile(),
            ConfigCommandAnalyze(),
            ConfigCommandReport()
        ]

        for config in config_array:
            for field in config.get_config():
                YamlConfigValidator._valid_yaml_options.add(field)

        YamlConfigValidator._initialized = True
