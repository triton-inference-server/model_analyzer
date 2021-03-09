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
        self._failing_configurations = {}
        self._passing_configurations = {}
        self._passing_measurements = {}
        self._failing_measurements = {}

    def total_configurations(self, model_name):
        """
        Parameters
        ----------
        model_name: str
            The name of the model for which
            we are getting the total number of 
            configurations

        Returns
        -------
        int 
            total number of configuration 
            searched by the model analyzer
            for a given model.
        """

        return self.passing_configurations(
            model_name) + self.failing_configurations(model_name)

    def set_passing_configurations(self, model_name, passing):
        """
        Sets the number of passing configuration 
        searched by the model analyzer.

        Parameters
        ----------
        model_name: str
            The name of the model for which
            we are setting the number of passing
            configurations
        passing : int
            The number of passing configs tried
        """

        self._passing_configurations[model_name] = passing

    def passing_configurations(self, model_name):
        """
        Parameters
        ----------
        model_name: str
            The name of the model for which
            we are getting the number of passing
            configurations

        Returns
        -------
        int 
            passing number of configurations 
            searched by the model analyzer.
        """

        return self._passing_configurations[model_name]

    def set_failing_configurations(self, model_name, failing):
        """
        Sets the number of failing configuration 
        searched by the model analyzer.

        Parameters
        ----------
         model_name: str
            The name of the model for which
            we are setting the number of failing
            configurations
        failing : int
            The total number of configs failing
        """

        self._failing_configurations[model_name] = failing

    def failing_configurations(self, model_name):
        """
        Parameters
        ----------
        model_name: str
            The name of the model for which
            we are getting the number of failing
            configurations

        Returns
        -------
        int 
            total number of failing configurations 
            searched by the model analyzer.
        """

        return self._failing_configurations[model_name]

    def total_measurements(self, model_name):
        """
        Parameters
        ----------
        model_name: str
            The name of the model for which
            we are getting the total number of 
            measurements

        Returns
        -------
        int 
            total number of measurements 
            searched by the model analyzer
            for a given model.
        """

        return self.passing_measurements(
            model_name) + self.failing_measurements(model_name)

    def set_passing_measurements(self, model_name, passing):
        """
        Sets the number of passing configuration 
        searched by the model analyzer.

        Parameters
        ----------
        model_name: str
            The name of the model for which
            we are setting the number of passing
            measurements
        passing : int
            The number of passing measurements tried
        """

        self._passing_measurements[model_name] = passing

    def passing_measurements(self, model_name):
        """
        Parameters
        ----------
        model_name: str
            The name of the model for which
            we are getting the number of passing
            measurements

        Returns
        -------
        int 
            passing number of measurements 
            searched by the model analyzer.
        """

        return self._passing_measurements[model_name]

    def set_failing_measurements(self, model_name, failing):
        """
        Sets the number of failing configuration 
        searched by the model analyzer.

        Parameters
        ----------
         model_name: str
            The name of the model for which
            we are setting the number of failing 
            measurements
        failing : int
            The total number of configs failing
        """

        self._failing_measurements[model_name] = failing

    def failing_measurements(self, model_name):
        """
        Parameters
        ----------
        model_name: str
            The name of the model for which
            we are getting the number of failing 
            measurements

        Returns
        -------
        int 
            total number of failing measurements 
            searched by the model analyzer.
        """

        return self._failing_measurements[model_name]
