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

from .result_utils import average_list
from model_analyzer.constants import COMPARISON_SCORE_THRESHOLD
from model_analyzer.result.measurement import Measurement


class ResultComparator:
    """
    Stores information needed to compare results and measurements.
    """

    def __init__(self, metric_objectives):
        """
        Parameters
        ----------
        metric_objectives : dict of RecordTypes
            keys are the metric types, and values are The relative importance
            of the keys with respect to other. If the values are 0,
        """

        # Normalize metric weights
        self._metric_weights = {
            key: (val / sum(metric_objectives.values()))
            for key, val in metric_objectives.items()
        }

    def compare_results(self, result1, result2):
        """
        Computes score for two results and compares
        the scores.

        Parameters
        ----------
        result1 : ModelResult
            first result to be compared
        result2 : ModelResult
            second result to be compared

        Returns
        -------
        int
            0
                if the results are determined
                to be the same within a threshold
            1
                if result1 > result2
            -1
                if result1 < result2
        """

        # For now create an average measurment
        agg_measurement1 = self._aggregate_measurements(
            result=result1, aggregation_func=average_list)
        agg_measurement2 = self._aggregate_measurements(
            result=result2, aggregation_func=average_list)

        return self.compare_measurements(measurement1=agg_measurement1,
                                         measurement2=agg_measurement2)

    def compare_measurements(self, measurement1, measurement2):
        """
        Compares individual meausurements retrieved from perf runs
        based on their scores

        Parameters
        ----------
        measurement1 : Measurement
            first set of (gpu_measurements, non_gpu_measurements) to
            be compared
        measurement2 : Measurement
            first set of (gpu_measurements, non_gpu_measurements) to
            be compared

        Returns
        -------
        int
            0
                if the results are determined
                to be the same within a threshold
            1
                if measurement1 > measurement2
            -1
                if measurement1 < measurement2
        """

        score_gain = 0.0
        for objective, weight in self._metric_weights.items():
            metric1 = measurement1.get_value_of_metric(tag=objective)
            metric2 = measurement2.get_value_of_metric(tag=objective)
            metric_diff = metric1 - metric2
            score_gain += (weight * (metric_diff.value() / metric1.value()))

        if score_gain > COMPARISON_SCORE_THRESHOLD:
            return 1
        elif score_gain < -COMPARISON_SCORE_THRESHOLD:
            return -1
        return 0

    def _aggregate_measurements(self, result, aggregation_func):
        """
        Returns
        -------
        (list, list)
            A 2-tuple of average measurements, 
            The first is across non-gpu specific metrics
            The second is across gpu-specific measurements
        """

        # For the gpu_measurements we have a list of dicts of lists
        # Assumption here is that its okay to average over all GPUs over all perf runs
        # This is done within the measurement itself

        measurements = result.measurements()
        aggregated_data = aggregation_func(
            [measurement.data() for measurement in measurements])

        # Data must be passed in as non-gpu-specific here.
        measurement = Measurement(gpu_data={},
                                  non_gpu_data=aggregated_data,
                                  perf_config=None)
        measurement.set_result_comparator(self)
        return measurement
