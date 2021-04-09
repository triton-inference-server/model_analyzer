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


class ConfigModel:
    """
    A class representing the configuration used for
    a single model.
    """
    def __init__(self,
                 model_name,
                 objectives=None,
                 constraints=None,
                 parameters=None,
                 model_config_parameters=None,
                 perf_analyzer_flags=None,
                 triton_server_flags=None):
        """
        Parameters
        ----------
        model_name : str
           Name used for the model
        objectives : None or list
           List of the objectives required by the model
        constraints : None or dict
            Constraints required by the model
        parameters : None or dict
            Constraints on batch_size and concurrency values need to be
            specified here.
        model_config_parameters : None or dict
            Model config parameters that is used for this model
        perf_analyzer_flags : dict
            The custom perf analyzer configuration 
            for this model
        triton_server_flags : dict
            The configuration for the triton server instance launched
            for this model
        """

        self._model_name = model_name
        self._objectives = objectives
        self._constraints = constraints
        self._parameters = parameters
        self._model_config_parameters = model_config_parameters
        self._perf_analyzer_flags = perf_analyzer_flags
        self._triton_server_flags = triton_server_flags

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

    def parameters(self):
        """
        Returns
        -------
        dict or None
            A dictionary containing the parameters
        """

        return self._parameters

    def model_config_parameters(self):
        """
        Returns
        -------
        dict or None
            A dictionary containing the model config parameters.
        """

        return self._model_config_parameters

    def model_name(self):
        """
        Returns
        -------
        str
            The model name used for this config.
        """

        return self._model_name

    def perf_analyzer_flags(self):
        """
        Returns
        -------
        dict:
             the perf_analyzer_flags
        """

        return self._perf_analyzer_flags

    def triton_server_flags(self):
        """
        Returns
        -------
        dict:
             the triton_server_flags
        """

        return self._triton_server_flags

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

    def set_parameters(self, parameters):
        """
        Parameters
        -------
        parameters : dict or None
            A dictionary containing the parameters
        """

        self._parameters = parameters

    def set_model_config_parameters(self, model_config_parameters):
        """
        Parameters
        ----------
        model_config_parameters : dict or None
            A dictionary containing the model config parameters.
        """

        self._model_config_parameters = model_config_parameters

    def set_model_name(self, model_name):
        """
        Parameters
        -------
        model_name : str
            The model name used for this config.
        """

        self._model_name = model_name

    def set_perf_analyzer_flags(self, flags):
        """
        Parameters
        -------
        flags: dict
             the perf_analyzer_flags
        """

        self._perf_analyzer_flags = flags

    def set_triton_server_flags(self, flags):
        """
        Parameters
        -------
        flags: dict
             the triton_server_flags
        """

        self._triton_server_flags = flags

    @staticmethod
    def model_object_to_config_model(value):
        """
        Converts a ConfigObject to ConfigModel.

        Parameters
        ----------
        value : dict
            A dictionary where the keys are model names
            and the values are ConfigObjects.

        Returns
        -------
        list
            A list of ConfigModel objects.
        """

        models = []
        for model_name, model_properties in value.items():
            models.append(ConfigModel(model_name, **model_properties.value()))

        return models

    @staticmethod
    def model_str_to_config_model(model_name):
        """
        Constructs a ConfigModel from a given
        model_name.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        ConfigModel
            ConfigModel object with the given model name.
        """

        return ConfigModel(model_name)

    @staticmethod
    def model_list_to_config_model(model_names):
        """
        Construct ConfigModel objects from a list of strings.

        Parameters
        ----------
        model_names : list
            A list of strings containing model names.

        Returns
        -------
        list
            A list of ConfigModel objects.
        """

        models = []
        for model_name in model_names:
            models.append(ConfigModel(model_name))
        return models

    @staticmethod
    def model_mixed_to_config_model(models):
        """
        Unifies a mixed list of ConfigModel objects
        and list of ConfigModel objects.

        Parameters
        ----------
        models : list
            A mixed list containing lists or ConfigModel objects.

        Returns
        -------
        list
            A list only containing ConfigModel objects.
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

        if self._parameters:
            model_object['parameters'] = self._parameters

        if self._constraints:
            model_object['constraints'] = self._constraints

        if self._model_config_parameters:
            model_object['model_config_parameters'] = \
                self._model_config_parameters

        if self._perf_analyzer_flags:
            model_object['perf_analyzer_flags'] = self._perf_analyzer_flags

        return str(model_object)
