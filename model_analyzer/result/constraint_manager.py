# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from typing import List, Dict, TYPE_CHECKING

from model_analyzer.record.record import Record

if TYPE_CHECKING:
    from model_analyzer.result.run_config_measurement import RunConfigMeasurement


class ConstraintManager:
    """
    Handles processing and applying
    constraints on a given measurements
    """

    @staticmethod
    def get_constraints_for_all_models(config):
        """
        Parameters
        ----------
        config :ConfigCommandProfile
            The model analyzer config

        Returns
        -------
        dict
            keys are model names, and values are constraints
        """

        constraints = {}
        for model in config.profile_models:
            constraints[model.model_name()] = model.constraints()
        if "constraints" in config.get_all_config():
            constraints["default"] = config.get_all_config()["constraints"]
        return constraints

    @staticmethod
    def satisfies_constraints(
            constraints: List[Dict[str, Dict[str, int]]],
            run_config_measurement: 'RunConfigMeasurement') -> bool:
        """
        Checks that the measurements, for every model, satisfy 
        the provided list of constraints

        Parameters
        ----------
        constraints: list of dicts
            keys are metrics and values are 
            constraint_type:constraint_value pairs
        run_config_measurement : RunConfigMeasurement
            The measurement to check against the constraints

        Returns
        -------
        True if measurement passes constraints
        False otherwise
        """

        if constraints:
            for (i, model_metrics) in enumerate(run_config_measurement.data()):
                for metric in model_metrics:
                    if ConstraintManager._metric_matches_constraint(
                            metric, constraints[i]):
                        if ConstraintManager._get_failure_percentage(
                                metric, constraints[i][metric.tag]) > 0:
                            return False

        return True

    @staticmethod
    def constraint_failure_percentage(
            constraints: List[Dict[str, Dict[str, int]]],
            run_config_measurement: 'RunConfigMeasurement') -> float:
        """
        Additive percentage, for every measurement, in every model, of how much 
        the RCM is failing the constraints by
        
        Returns
        -------
        float
        """
        failure_percentage: float = 0

        if constraints:
            for (i, model_metrics) in enumerate(run_config_measurement.data()):
                for metric in model_metrics:
                    if ConstraintManager._metric_matches_constraint(
                            metric, constraints[i]):
                        failure_percentage += ConstraintManager._get_failure_percentage(
                            metric, constraints[i][metric.tag])

        return failure_percentage * 100

    @staticmethod
    def _metric_matches_constraint(
            metric: Record, constraint: Dict[str, Dict[str, int]]) -> bool:
        if constraint is not None and metric.tag in constraint:
            return True
        else:
            return False

    @staticmethod
    def _get_failure_percentage(metric: Record, constraint: Dict[str,
                                                                 int]) -> float:

        failure_percentage = 0

        if 'min' in constraint:
            if metric.value() < constraint['min']:
                failure_percentage = (constraint['min'] -
                                      metric.value()) / constraint['min']
        if 'max' in constraint:
            if metric.value() > constraint['max']:
                failure_percentage = (metric.value() -
                                      constraint['max']) / constraint['max']

        return failure_percentage