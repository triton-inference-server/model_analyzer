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

from typing import List, Dict


class RunConfigResultComparator:
    """
    Stores information needed to compare two RunConfigResults.
    """

    def __init__(self, metric_objectives_list: List[Dict[str, int]],
                 model_weights: List[int]):
        """
        Parameters
        ----------
        List of
            metric_objectives : dict of RecordTypes
                keys are the metric types, and values are The relative importance
                of the keys with respect to other. If the values are 0,
        """

        # Normalize metric weights
        self._metric_weights = []
        self._model_weights = []
        for i, metric_objectives in enumerate(metric_objectives_list):
            self._metric_weights.append({
                key: (val / sum(metric_objectives.values()))
                for key, val in metric_objectives.items()
            })

            self._model_weights.append(model_weights[i])

    def get_metric_weights(self):
        return self._metric_weights

    def get_model_weights(self):
        return self._model_weights

    def is_better_than(self, run_config_result1, run_config_result2):
        """
        Aggregates and compares the score for two RunConfigResults 

        Parameters
        ----------
        run_config_result1 : RunConfigResult
            first result to be compared
        run_config_result2 : RunConfigResult
            second result to be compared

        Returns
        -------
        bool
           True: if result1 is better than result2
        """

        agg_run_config_measurement1 = self._aggregate_run_config_measurements(
            run_config_result1, aggregation_func=max)
        agg_run_config_measurement2 = self._aggregate_run_config_measurements(
            run_config_result2, aggregation_func=max)

        return agg_run_config_measurement1.is_better_than(
            agg_run_config_measurement2)

    def _aggregate_run_config_measurements(self, run_config_result,
                                           aggregation_func):
        """
        Returns
        -------
        (list, list)
            A 2-tuple of average RunConfigMeasurements, 
            The first is across non-gpu specific metrics
            The second is across gpu-specific measurements
        """

        # For the gpu_measurements we have a list of dicts of lists
        # Assumption here is that its okay to average over all GPUs over all perf runs
        # This is done within the measurement itself

        if run_config_result.passing_measurements():
            aggregated_run_config_measurement = aggregation_func(
                run_config_result.passing_measurements())
        else:
            aggregated_run_config_measurement = aggregation_func(
                run_config_result.run_config_measurements())

        aggregated_run_config_measurement.set_model_config_weighting(
            self._model_weights)
        aggregated_run_config_measurement.set_metric_weightings(
            self._metric_weights)

        return aggregated_run_config_measurement
