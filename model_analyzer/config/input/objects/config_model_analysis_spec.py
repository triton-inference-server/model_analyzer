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


class ConfigModelAnalysisSpec:
    """
    A class representing the configuration used for
    a single model.
    """

    def __init__(self, model_name, objectives=None, constraints=None):
        """
        Parameters
        ----------
        model_name : str
           Name used for the model
        objectives : None or list
           List of the objectives required by the model
        constraints : None or dict
            Constraints required by the model
        """

        self._model_name = model_name
        self._objectives = objectives
        self._constraints = constraints

    def objectives(self):
        """
        Returns
        -------
        list or None
            A list containing the objectives.
        """

        return self._objectives

    def constraints(self):
        """
        Returns
        -------
        dict or None
            A dictionary containing the constraints
        """

        return self._constraints

    def model_name(self):
        """
        Returns
        -------
        str
            The model name used for this config.
        """

        return self._model_name

    def set_objectives(self, objectives):
        """
        Parameters
        -------
        objectives : dict or None
            A dictionary containing the parameters
        """

        self._objectives = objectives

    def set_constraints(self, constraints):
        """
        Parameters
        -------
        constraints : dict or None
            A dictionary containing the parameters
        """

        self._constraints = constraints

    def set_model_name(self, model_name):
        """
        Parameters
        -------
        model_name : str
            The model name used for this config.
        """

        self._model_name = model_name

    @staticmethod
    def model_object_to_config_model_analysis_spec(value):
        """
        Converts a ConfigObject to ConfigModelAnalysisSpec.

        Parameters
        ----------
        value : dict
            A dictionary where the keys are model names
            and the values are ConfigObjects.

        Returns
        -------
        list
            A list of ConfigModelAnalysisSpec objects.
        """

        models = []
        for model_name, model_properties in value.items():
            models.append(
                ConfigModelAnalysisSpec(model_name, **model_properties.value()))

        return models

    @staticmethod
    def model_str_to_config_model_analysis_spec(model_name):
        """
        Constructs a ConfigModelAnalysisSpec from a given
        model_name.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        ConfigModelAnalysisSpec
            ConfigModelAnalysisSpec object with the given model name.
        """

        return ConfigModelAnalysisSpec(model_name)

    @staticmethod
    def model_list_to_config_model_analysis_spec(analysis_models):
        """
        Construct ConfigModelAnalysisSpec objects from a list of strings.

        Parameters
        ----------
        analysis_models : list
            A list of strings containing model names.

        Returns
        -------
        list
            A list of ConfigModelAnalysisSpec objects.
        """

        models = []
        for model_name in analysis_models:
            models.append(ConfigModelAnalysisSpec(model_name))
        return models

    @staticmethod
    def model_mixed_to_config_model_analysis_spec(models):
        """
        Unifies a mixed list of ConfigModelAnalysisSpec objects
        and list of ConfigModelAnalysisSpec objects.

        Parameters
        ----------
        models : list
            A mixed list containing lists or ConfigModelAnalysisSpec objects.

        Returns
        -------
        list
            A list only containing ConfigModelAnalysisSpec objects.
        """

        new_models = []
        for model in models:
            if type(model.value()) is list:
                for model_i in model.value():
                    new_models.append(model_i)
            else:
                new_models.append(model.value())
        return new_models

    def __repr__(self):
        model_object = {'model_name': self._model_name}
        if self._objectives:
            model_object['objectives'] = self._objectives

        if self._constraints:
            model_object['constraints'] = self._constraints

        return str(model_object)
