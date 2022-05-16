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

logger = logging.getLogger(LOGGER_NAME)


class YamlConfigValidator:
    """
    Validate all options from the yaml file.
    """

    def __init__(self):
        """
        Create set of valid yaml options
        """
        self._valid_yaml_options = {
            "profile_models",
            "constraints",
            "objectives",
            "triton_server_flags",
            "perf_analyzer_flags",
            "triton_server_environment",
            "triton_docker_labels",
            "analysis_models",
            "constraints",
            "objectives",
            "report_model_configs",
            "plots",
        }
        self._add_config_valid_options()

    def _add_config_valid_options(self):
        # Importing here to remove a circular dependency.
        # If these imports are moved to the top of the file, they are imported at file load.
        # This will cause a circular dependency between ConfigCommand, ConfigCommand*, and YamlConfigValidator
        # However, importing here, only requires these files to be imported on class initialization.
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
                self._valid_yaml_options.add(field)

    def is_valid_option(self, option):
        if option not in self._valid_yaml_options:
            logger.error(
                f'{option} is not a valid yaml argument. Please be sure to use underscores and check spelling.'
            )
            return False
        return True
