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

from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from model_analyzer.constants import LOGGER_NAME

import logging

logger = logging.getLogger(LOGGER_NAME)


class Results:
    """
    Provides storage and accessor functions for measurements
    """
    RUN_CONFIG_INDEX = 0
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
        {models_name: { [run_config: (key, {run_config_measurements} ) ] } }
        ---results_dict-------------------------------------------
                     ---model_dict------------------------------
                       ---model_config_tuple_list-------------
                                      -key, run_config_measurement_dict-
        """
        results = Results()

        for models_name, model_dict in results_dict['_results'].items():
            for model_variants_name, run_config_tuple_list in model_dict.items(
            ):
                run_config = RunConfig.from_dict(
                    run_config_tuple_list[Results.RUN_CONFIG_INDEX])

                for key, measurement_dict in run_config_tuple_list[
                        Results.MEASUREMENTS_INDEX].items():
                    run_config_measurement = RunConfigMeasurement.from_dict(
                        measurement_dict)

                    results._add_run_config_measurement(models_name, run_config,
                                                        model_variants_name,
                                                        key,
                                                        run_config_measurement)

        return results

    def add_run_config_measurement(self, run_config, run_config_measurement):
        """
        Given a ModelRunConfig and a RunConfigMeasurement, add the measurement to the
        ModelRunConfig's measurements
        
        Parameters
        ----------
        run_config: RunConfig
        key: str
        run_config_measurement: RunConfigMeasurement
        """

        models_name = run_config.models_name()
        model_variants_name = run_config.model_variants_name()
        key = run_config.representation()

        self._add_run_config_measurement(models_name, run_config,
                                         model_variants_name, key,
                                         run_config_measurement)

    def contains_model(self, models_name):
        """
        Checks if the models name exists
        in the results

        Parameters
        ----------
        models_name : str
            The models name

        Returns
        -------
        bool
        """
        return models_name in self._results

    def contains_model_variant(self, models_name, model_variants_name):
        """
        Checks if the models name and model variants name
        exist in the results
        
        Parameters
        ----------
        models_name : str
        model_variants_name: str
            
        Returns
        -------
        bool
        """
        if not self.contains_model(models_name):
            return False

        return model_variants_name in self._results[models_name]

    def get_list_of_models(self):
        """
        Returns the list of models profiled
        
        Returns
        -------
        list of str
            List of the names of model's profiled
        """
        return list(self._results.keys())

    def get_list_of_model_config_measurement_tuples(self):
        """
        Returns a list of model configs with their 
        corresponding RunConfigMeasurements
        
        Returns
        -------
        list of tuples - [(ModelConfig, list of RunConfigMeasurements)]
            List of model configs and a dict of all associated
            measurement values
        """
        return list(self._results.values())

    def get_list_of_run_config_measurements(self):
        """
        Return a list of RunConfigMeasurements from every model/model_config
        
        Parameters
        ----------
        None
        
        Returns
        -------
        List of RunConfigMeasurements
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
        # FIXME is this description correct?
        Given a model name, return a dict of tuples of model configs and 
        a dict of all associated measurement values
        
        Parameters
        ----------
        model_name : str
            The model name for the requested results
            
        Returns
        -------
        Dict of tuples - {(ModelConfig, list of RunConfigMeasurements)}
            Dict of tuples consisting of the model's ModelConfig 
            and a list of all associated measurement values
        """
        if not self.contains_model(model_name):
            logger.error(f'No results found for model: {model_name}')
            return {}

        return self._results[model_name]

    def get_model_variants_measurements_dict(self, models_name,
                                             model_variants_name):
        """
        FIXME: is this description correct?
        Given a models name and model variants name, return a dict where the
        key is run_config and the values are list of all associated measurements
        
        Parameters
        ----------
        model_name : str
        model_variants_name: str

        Returns
        -------
        Dict of RunConfigMeasurements
        """
        if not self.contains_model(
                models_name) or not self.contains_model_variant(
                    models_name, model_variants_name):
            logger.error(f'No results found for variant: {model_variants_name}')
            return {}

        return self._results[models_name][model_variants_name][
            Results.MEASUREMENTS_INDEX]

    def get_all_model_variant_measurements(self, model_name,
                                           model_variant_name):
        """
        Given a model name and model variant name, return the RunConfig and 
        a list of all associated measurement values
        
        Parameters
        ----------
        model_name : str
        model_variant_name: str

        Returns
        -------
        Tuple - (RunConfig, list of RunConfigMeasurements)
            Tuple consisting of the model_config and a list of
            all associated measurement values
        """
        # FIXME -- run config or model config here?
        if not self.contains_model(
                model_name) or not self.contains_model_variant(
                    model_name, model_variant_name):
            logger.error(
                f'No results found for model config: {model_variant_name}')
            return (None, [])

        model_config_data = self._results[model_name][model_variant_name]

        return model_config_data[Results.RUN_CONFIG_INDEX], list(
            model_config_data[Results.MEASUREMENTS_INDEX].values())

    def _add_run_config_measurement(self, models_name, run_config,
                                    model_variants_name, key,
                                    run_config_measurement):
        if models_name not in self._results:
            self._results[models_name] = {}

        if model_variants_name not in self._results[models_name]:
            self._results[models_name][model_variants_name] = (run_config, {})

        self._results[models_name][model_variants_name][
            Results.MEASUREMENTS_INDEX][key] = run_config_measurement
