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

from typing import Union, Dict, TYPE_CHECKING

from model_analyzer.record.record import Record

if TYPE_CHECKING:
    from model_analyzer.result.run_config_measurement import RunConfigMeasurement

from model_analyzer.constants import GLOBAL_CONSTRAINTS_KEY
from model_analyzer.result.model_constraints import ModelConstraints
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_report import ConfigCommandReport


class ConstraintManager:
    """
    Handles processing and applying
    constraints on a given measurements

    Parameters
    ----------
    config: ConfigCommandProfile or ConfigCommandReport
    """

    def __init__(
            self, config: Union[ConfigCommandProfile,
                                ConfigCommandReport]) -> None:
        self._constraints = {}

        if config:
            # Model constraints
            if "profile_models" in config.get_config():
                for model in config.profile_models:
                    self._constraints[model.model_name()] = model.constraints()

            # Global constraints
            if "constraints" in config.get_all_config():
                self._constraints[GLOBAL_CONSTRAINTS_KEY] = ModelConstraints(
                    config.get_all_config()["constraints"])

    def get_constraints_for_all_models(self):
        """
        Returns
        -------
        dict
            keys are model names, and values are ModelConstraints objects
        """

        return self._constraints

    def satisfies_constraints(
            self, run_config_measurement: 'RunConfigMeasurement') -> bool:
        """
        Checks that the measurements, for every model, satisfy 
        the provided list of constraints

        Parameters
        ----------
        run_config_measurement : RunConfigMeasurement
            The measurement to check against the constraints

        Returns
        -------
        True if measurement passes constraints
        False otherwise
        """

        if self._constraints:
            for (model_name,
                 model_metrics) in run_config_measurement.data().items():
                for metric in model_metrics:
                    if self._metric_matches_constraint(
                            metric, self._constraints[model_name]):
                        if self._get_failure_percentage(
                                metric,
                                self._constraints[model_name][metric.tag]) > 0:
                            return False

        return True

    def constraint_failure_percentage(
            self, run_config_measurement: 'RunConfigMeasurement') -> float:
        """
        Additive percentage, for every measurement, in every model, of how much 
        the RCM is failing the constraints by
        
        Returns
        -------
        float
        """
        failure_percentage: float = 0

        if self._constraints:
            for (model_name,
                 model_metrics) in run_config_measurement.data().items():
                for metric in model_metrics:
                    if self._metric_matches_constraint(
                            metric, self._constraints[model_name]):
                        failure_percentage += self._get_failure_percentage(
                            metric, self._constraints[model_name][metric.tag])

        return failure_percentage * 100

    def _metric_matches_constraint(self, metric: Record,
                                   constraint: ModelConstraints) -> bool:
        if constraint.has_metric(metric.tag):
            return True
        else:
            return False

    def _get_failure_percentage(self, metric: Record,
                                constraint: Dict[str, int]) -> float:

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
