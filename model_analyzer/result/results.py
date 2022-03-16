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
        self._done = {}

    @classmethod
    def from_dict(cls, results_dict):
        results = Results()
        results._results = results_dict['_results']

        return results

    def add_measurement(self, run_config, key, measurement):
        """
        Given a RunConfig and a key, store a measurement
        
        Parameters
        ----------
        run_config: RunConfig
        key: str
        measurement: Measurement
        """
        model_name, model_config, model_config_name = self._extract_run_config_fields(
            run_config)

        if model_name not in self._results:
            self._results[model_name] = {}
            self._done[model_name] = False

        if model_config_name not in self._results[model_name]:
            self._results[model_name][model_config_name] = (model_config, {})

        self._results[model_name][model_config_name][
            Results.MEASUREMENTS_INDEX][key] = measurement

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
        return model_config_name in self._results[model_name]

    def is_done(self, model_name):
        return self._done[model_name]

    def next_result(self, model_name):
        """
        Given a model name, create a generator that returns
        model configs with their corresponding measurements
        
        Parameters
        ----------
        model_name : str
            The model name for the requested results

        Returns
        -------
        Tuple - (model_config_name, dict of measurements)
            Tuple consisting of the model_config_name and 
            it's corresponding measurements
        """
        if model_name not in self._results:
            logger.warning(
                f"Model {model_name} requested for analysis but no results were found. "
                "Ensure that this model was actually profiled.")
            return None

        results_list = list(self._results[model_name].values())

        self._done[model_name] = False
        for result in results_list:
            if result == results_list[-1]:
                self._done[model_name] = True

            yield result[Results.MODEL_CONFIG_INDEX], result[
                Results.MEASUREMENTS_INDEX]

    def get_all_model_measurements(self, model_name):
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

    def _extract_run_config_fields(self, run_config):
        return (run_config.model_name(), run_config.model_configs()[0],
                run_config.model_configs()[0].get_field('name'))
