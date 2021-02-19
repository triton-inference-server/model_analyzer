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
from model_analyzer.record.measurement import Measurement
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from collections import defaultdict


class ResultComparator:
    """
    Stores information needed
    to compare results and 
    measurements.
    """

    def __init__(self, gpu_metric_types, non_gpu_metric_types,
                 metric_objectives):
        """
        Parameters
        ----------
        gpu_metric_types : list of RecordTypes
            The types of measurements in the measurements
            list that have a GPU ID, in the order they appear
        non_gpu_metric_types : list of RecordTypes
            The types of measurements in the measurements
            list that do NOT have a GPU ID, in the order they appear
        metric_objectives : dict of RecordTypes
            keys are the metric types, and values are The relative importance 
            of the keys with respect to other. If the values are 0,
        """

        # Index the non gpu metric types
        self._non_gpu_type_to_idx = {
            non_gpu_metric_types[i]: i
            for i in range(len(non_gpu_metric_types))
        }

        # Index the gpu metric types
        self._gpu_type_to_idx = {
            gpu_metric_types[i]: i
            for i in range(len(gpu_metric_types))
        }

        # Normalize metric objectives
        self._metric_objectives = {
            key: (val / sum(metric_objectives.values()))
            for key, val in metric_objectives.items()
        }

    def compare_results(self, result1, result2):
        """
        Computes score for two results and compares
        the scores.

        Parameters
        ----------
        result1 : RunResult
            first result to be compared
        result2 : RunResult
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

        gpu_data1 = measurement1.gpu_data()
        gpu_data2 = measurement2.gpu_data()

        gpu_rows1 = []
        gpu_rows2 = []

        for gpu_row in gpu_data1.values():
            gpu_rows1.append(gpu_row)
        for gpu_row in gpu_data2.values():
            gpu_rows2.append(gpu_row)

        avg_gpu_data1 = average_list(gpu_rows1)
        avg_gpu_data2 = average_list(gpu_rows2)

        non_gpu_data1 = measurement1.non_gpu_data()
        non_gpu_data2 = measurement2.non_gpu_data()

        score_diff = 0.0
        for objective, weight in self._metric_objectives.items():
            if objective in self._non_gpu_type_to_idx:
                metric_idx = self._non_gpu_type_to_idx[objective]
                value_diff = (non_gpu_data1[metric_idx] -
                              non_gpu_data2[metric_idx]).value()
                value_sum = (non_gpu_data1[metric_idx] +
                             non_gpu_data2[metric_idx]).value()
            elif objective in self._gpu_type_to_idx:
                metric_idx = self._gpu_type_to_idx[objective]
                value_diff = (avg_gpu_data1[metric_idx] -
                              avg_gpu_data2[metric_idx]).value()
                value_sum = (avg_gpu_data1[metric_idx] +
                             avg_gpu_data2[metric_idx]).value()
            else:
                raise TritonModelAnalyzerException(
                    f"Category unknown for objective : '{objective}'")
            score_diff += (weight * (value_diff / value_sum))

        # Compare score with threshhold
        if score_diff > COMPARISON_SCORE_THRESHOLD:
            return 1
        elif score_diff < -COMPARISON_SCORE_THRESHOLD:
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

        measurements = result.get_measurements()

        gpu_data = defaultdict(list)
        non_gpu_rows = []
        for measurement in measurements:
            gpu_measurement_data = measurement.gpu_data()
            for gpu_id, gpu_row in gpu_measurement_data.items():
                gpu_data[gpu_id].append(gpu_row)
            non_gpu_rows.append(measurement.non_gpu_data())

        # Aggregate the data
        for gpu_id, gpu_rows in gpu_data.items():
            gpu_data[gpu_id] = aggregation_func(gpu_rows)
        non_gpu_data = aggregation_func(non_gpu_rows)

        return Measurement(gpu_data=gpu_data,
                           non_gpu_data=non_gpu_data,
                           perf_config=None,
                           comparator=self)
