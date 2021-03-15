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
        config : AnalyzerConfig
            The model analyzer config

        Returns
        -------
        dict
            keys are model names, and values are constraints
        """

        constraints = {}
        for model in config.model_names:
            constraints[model.model_name()] = model.constraints()
        return constraints

    @staticmethod
    def check_constraints(constraints, measurement):
        """
        Takes a measurement and
        checks it against the constraints.

        Parameters
        ----------
        constraints: dict
            keys are metrics and values are 
            constraint_type:constraint_value pairs
        measurement : list of metrics
            The measurement to check against the constraints

        Return
        ------
        True if measurement passes constraints
        False otherwise
        """

        if constraints:
            for metric in measurement.data():
                if type(metric).tag in constraints:
                    constraint = constraints[type(metric).tag]
                    if 'min' in constraint:
                        if metric.value() < constraint['min']:
                            return False
                    if 'max' in constraint:
                        if metric.value() > constraint['max']:
                            return False
        return True
