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


#TODO-TMA-573: This class has no unit testing
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
        for model in config.analysis_models:
            constraints[model.model_name()] = model.constraints()
        if "constraints" in config.get_all_config():
            constraints["default"] = config.get_all_config()["constraints"]
        return constraints

    @staticmethod
    def satisfies_constraints(constraints, run_config_measurement):
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

        Return
        ------
        True if measurement passes constraints
        False otherwise
        """

        if constraints:
            for (i, model_metrics) in enumerate(run_config_measurement.data()):
                for metric in model_metrics:
                    if constraints[i] is not None and type(
                            metric).tag in constraints[i]:
                        constraint = constraints[i][type(metric).tag]
                        if 'min' in constraint:
                            if metric.value() < constraint['min']:
                                return False
                        if 'max' in constraint:
                            if metric.value() > constraint['max']:
                                return False

        return True
