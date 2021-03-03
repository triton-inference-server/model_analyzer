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


class AnalyzerStatistics:
    """
    An object containing all statistics relevant
    to an instance of Analyzer
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config: AnalyzerConfig
            The config for this model analyzer
        """

        self._config = config
        self._failing_configurations = 0
        self._passing_configurations = 0

    def total_configurations(self):
        """
        Returns
        -------
        int 
            total number of configuration 
            searched by the model analyzer.
        """

        return self._passing_configurations + self._failing_configurations

    def set_passing_configurations(self, passing):
        """
        Sets the number of passing configuration 
        searched by the model analyzer.

        Parameters
        ----------
        total : int
            The total number of configs tried
        """

        self._passing_configurations = passing

    def passing_configurations(self):
        """
        Returns
        -------
        int 
            total number of configuration 
            searched by the model analyzer.
        """

        return self._passing_configurations

    def set_failing_configurations(self, failing):
        """
        Sets the number of failing configuration 
        searched by the model analyzer.

        Parameters
        ----------
        total : int
            The total number of configs tried
        """

        self._failing_configurations = failing

    def failing_configurations(self):
        """
        Returns
        -------
        int 
            total number of failing configurations 
            searched by the model analyzer.
        """

        return self._failing_configurations
