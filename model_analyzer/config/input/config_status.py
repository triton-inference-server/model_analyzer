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


class ConfigStatus:

    def __init__(self, status, message=None, config_object=None):
        """
        Create a new ConfigStatus

        Parameters
        ----------
        status : int
            Status of the config parsing. Accepted
            values are CONFIG_PARSER_SUCCESS and
            CONFIG_PARSER_FAILURE.

        message : str
            A string message containing the description.

        config_object : ConfigValue
            ConfigObject that is creating this status.
        """

        self._status = status
        self._message = message
        self._config_object = config_object

    def status(self):
        """
        Get the config status.

        Returns
        -------
        int
            Config status
        """

        return self._status

    def message(self):
        """
        Get the message.

        Returns
        -------
        str
            The message for the status.
        """

        return self._message

    def config_object(self):
        """
        Get the config object for this status.

        Returns
        -------
        ConfigValue or None
            Config object that created this status.
        """

        return self._config_object
