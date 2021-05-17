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


class ConfigModelReportSpec:
    """
    A class representing the configuration used for
    a single model.
    """

    def __init__(self, model_config_name, plots=None):
        """
        Parameters
        ----------
        model_config_name : str
           Name used for the model
        plots : list of ConfigPlot
            containing plot names and ConfigObjects
        """

        self._model_config_name = model_config_name
        self._plots = plots

    def model_config_name(self):
        """
        Returns
        -------
        str
            The model name used for this config.
        """

        return self._model_config_name

    def plots(self):
        """
        Returns
        -------
        list of ConfigPlot
            containing plot names and ConfigObjects
        """

        return self._plots

    def set_model_config_name(self, model_config_name):
        """
        Parameters
        -------
        model_config_name : str
            The model name used for this config.
        """

        self._model_config_name = model_config_name

    def set_plots(self, plots):
        """
        Parameters
        ----------
        plots : list of ConfigPLot
            containing plot names and ConfigObjects
        """

        self._plots = plots

    @staticmethod
    def model_object_to_config_model_report_spec(value):
        """
        Converts a ConfigObject to ConfigModelReportSpec.

        Parameters
        ----------
        value : dict
            A dictionary where the keys are model names
            and the values are ConfigObjects.

        Returns
        -------
        list
            A list of ConfigModelReportSpec objects.
        """

        models = []
        for model_config_name, model_properties in value.items():
            models.append(
                ConfigModelReportSpec(model_config_name,
                                      **model_properties.value()))

        return models

    @staticmethod
    def model_str_to_config_model_report_spec(model_config_name):
        """
        Constructs a ConfigModelReportSpec from a given
        model_config_name.

        Parameters
        ----------
        model_config_name : str
            Name of the model

        Returns
        -------
        ConfigModelReportSpec
            ConfigModelReportSpec object with the given model name.
        """

        return ConfigModelReportSpec(model_config_name)

    @staticmethod
    def model_list_to_config_model_report_spec(report_models):
        """
        Construct ConfigModelReportSpec objects from a list of strings.

        Parameters
        ----------
        report_models : list
            A list of strings containing model names.

        Returns
        -------
        list
            A list of ConfigModelReportSpec objects.
        """

        models = []
        for model_config_name in report_models:
            models.append(ConfigModelReportSpec(model_config_name))
        return models

    @staticmethod
    def model_mixed_to_config_model_report_spec(models):
        """
        Unifies a mixed list of ConfigModelReportSpec objects
        and list of ConfigModelReportSpec objects.

        Parameters
        ----------
        models : list
            A mixed list containing lists or ConfigModelReportSpec objects.

        Returns
        -------
        list
            A list only containing ConfigModelReportSpec objects.
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
        model_object = {'model_config_name': self._model_config_name}
        if self._plots:
            model_object['plots'] = self._plots

        return str(model_object)
