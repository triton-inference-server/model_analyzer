# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.result.measurement import Measurement
from model_analyzer.constants import LOGGER_NAME

import logging

logger = logging.getLogger(LOGGER_NAME)


class Results:
    """
    Provides storage and accessor functions for measurements
    """
    MODEL_CONFIG_INDEX = 0
    MEASUREMENTS_INDEX = 1

    def __init__(self):
        """          
        """
        self._results = {}

    @classmethod
    def from_dict(cls, results_dict):
        """
        Populate the Results class based on the dictionary value
        (stored in the checkpoint)
        
        The checkpoint format is:
        {model_name: { [model_config: (key, {measurements} ) ] } }
        ---results_dict-------------------------------------------
                     ---model_dict------------------------------
                       ---model_config_tuple_list-------------
                                      -key, measurement_dict-
        """
        results = Results()

        for model_name, model_dict in results_dict['_results'].items():
            for model_config_name, model_config_tuple_list in model_dict.items(
            ):
                model_config = ModelConfig.from_dict(
                    model_config_tuple_list[Results.MODEL_CONFIG_INDEX])

                for key, measurement_dict in model_config_tuple_list[
                        Results.MEASUREMENTS_INDEX].items():
                    measurement = Measurement.from_dict(measurement_dict)

                    results._add_measurement(model_name, model_config,
                                             model_config_name, key,
                                             measurement)

        return results

    def add_measurement(self, model_run_config, key, measurement):
        """
        Given a RunConfig and a key, store a measurement
        
        Parameters
        ----------
        model_run_config: ModelRunConfig
        key: str
        measurement: Measurement
        """
        model_name, model_config, model_config_name = self._extract_model_run_config_fields(
            model_run_config)

        self._add_measurement(model_name, model_config, model_config_name, key,
                              measurement)

    def contains_model(self, model_name):
        """
        Checks if the model name exists
        in the results

        Parameters
        ----------
        model_name : str
            The model name

        Returns
        -------
        bool
        """
        return model_name in self._results

    def contains_model_config(self, model_name, model_config_name):
        """
        Checks if the model name and model config name
        exists in the results
        
        Parameters
        ----------
        model_name : str
            The model name
            
        model_config_name: str
            The model config name 
            
        Returns
        -------
        bool
        """
        if not self.contains_model(model_name):
            return False

        return model_config_name in self._results[model_name]

    def get_list_of_models(self):
        """
        Returns the list of models profiled
        
        Returns
        -------
        list of str
            List of the names of model's profiled
        """
        return list(self._results.keys())

    def get_list_of_model_config_measurements(self):
        """
        Returns a list of model config measurements
        
        Returns
        -------
        list of tuples - [(ModelConfig, list of Measurements)]
            List of model configs and a dict of all associated
            measurement values
        """
        return list(self._results.values())

    def get_list_of_measurements(self):
        """
        Return a list of measurements from every model/model_config
        
        Parameters
        ----------
        None
        
        Returns
        -------
        List of Measurements
        """
        measurements = []
        for model_result in self._results.values():
            for model_config_result in model_result.values():
                for measurement in model_config_result[
                        Results.MEASUREMENTS_INDEX].values():
                    measurements.append(measurement)

        return measurements

    def get_model_measurements_dict(self, model_name):
        """
        Given a model name, return a dict of tuples of model configs and 
        a dict of all associated measurement values
        
        Parameters
        ----------
        model_name : str
            The model name for the requested results
            
        Returns
        -------
        Dict of tuples - {(ModelConfig, list of Measurements)}
            Dict of tuples consisting of the model's ModelConfig 
            and a list of all associated measurement values
        """
        if not self.contains_model(model_name):
            logger.error(f'No results found for model: {model_name}')
            return {(None, [])}

        return self._results[model_name]

    def get_model_config_measurements_dict(self, model_name, model_config_name):
        """
        Given a model name and model config name, return a dict where the
        key is model configs and the values are list of all associated measurements
        
        Parameters
        ----------
        model_name : str
            The model name for the requested results
            
        model_config_name: str
            The model config name for the requested results

        Returns
        -------
        Dict of Measurements
        """
        if not self.contains_model(
                model_name) or not self.contains_model_config(
                    model_name, model_config_name):
            logger.error(
                f'No results found for model config: {model_config_name}')
            return {}

        return self._results[model_name][model_config_name][
            Results.MEASUREMENTS_INDEX]

    def get_all_model_config_measurements(self, model_name, model_config_name):
        """
        Given a model name and model config name, return the model config and 
        a list of all associated measurement values
        
        Parameters
        ----------
        model_name : str
            The model name for the requested results
            
        model_config_name: str
            The model config name for the requested results

        Returns
        -------
        Tuple - (ModelConfig, list of Measurements)
            Tuple consisting of the model_config and a list of
            all associated measurement values
        """
        if not self.contains_model(
                model_name) or not self.contains_model_config(
                    model_name, model_config_name):
            logger.error(
                f'No results found for model config: {model_config_name}')
            return (None, [])

        model_config_data = self._results[model_name][model_config_name]

        return model_config_data[Results.MODEL_CONFIG_INDEX], list(
            model_config_data[Results.MEASUREMENTS_INDEX].values())

    def _add_measurement(self, model_name, model_config, model_config_name, key,
                         measurement):
        if model_name not in self._results:
            self._results[model_name] = {}

        if model_config_name not in self._results[model_name]:
            self._results[model_name][model_config_name] = (model_config, {})

        self._results[model_name][model_config_name][
            Results.MEASUREMENTS_INDEX][key] = measurement

    def _extract_model_run_config_fields(self, model_run_config):
        return (model_run_config.model_name(), model_run_config.model_config(),
                model_run_config.model_config().get_field('name'))
