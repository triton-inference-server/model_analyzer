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
from model_analyzer.record.measurement import Measurement
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class ResultComparator:
    """
    Stores information needed
    to compare results and 
    measurements.
    """

    def __init__(self,
                 gpu_metric_types,
                 non_gpu_metric_types,
                 metric_priorities,
                 comparison_threshold_percent=1,
                 weights=None):
        """
        Parameters
        ----------
        gpu_metric_types : list of RecordTypes
            The types of measurements in the measurements
            list that have a GPU ID, in the order they appear
        non_gpu_metric_types : list of RecordTypes
            The types of measurements in the measurements
            list that do NOT have a GPU ID, in the order they appear
        metric_priorities : list of RecordTypes
            The priority of the types above (i.e. the order
            of comparison)
        comparison_threshold_percent : int
            The threshold within with two measurements are considered
            equal as a percentage of the first measurement
        weights : list of float
            The relative importance of the priorities with respect to others
            If None, then equal weighting (normal priority comparison)
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

        self._metric_priorities = metric_priorities
        self._comparison_threshold_factor = (comparison_threshold_percent *
                                             1.0) / 100

        # TODO implement a weighted comparison
        self._weights = weights

    def compare_results(self, result1, result2):
        """
        Compares two results using priorities 
        specified in model analyzer config

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
        based on priorities specified by model analyzer config

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

        for _, gpu_row in gpu_data1.items():
            gpu_rows1.append(gpu_row)
        for _, gpu_row in gpu_data2.items():
            gpu_rows2.append(gpu_row)

        avg_gpu_data1 = average_list(gpu_rows1)
        avg_gpu_data2 = average_list(gpu_rows2)

        non_gpu_data1 = measurement1.non_gpu_data()
        non_gpu_data2 = measurement2.non_gpu_data()

        for priority in self._metric_priorities:
            if priority in self._non_gpu_type_to_idx:
                # Get position in measurements of current priority's value
                metric_idx = self._non_gpu_type_to_idx[priority]
                threshold = self._comparison_threshold_factor * non_gpu_data1[
                    metric_idx].value()
                value_diff = (non_gpu_data1[metric_idx] -
                              non_gpu_data2[metric_idx]).value()

                if value_diff > threshold:
                    return 1
                elif value_diff < -threshold:
                    return -1
            elif priority in self._gpu_type_to_idx:
                metric_idx = self._gpu_type_to_idx[priority]
                threshold = self._comparison_threshold_factor * avg_gpu_data1[
                    metric_idx].value()
                value_diff = (avg_gpu_data1[metric_idx] -
                              avg_gpu_data2[metric_idx]).value()

                if value_diff > threshold:
                    return 1
                elif value_diff < -threshold:
                    return -1
            else:
                raise TritonModelAnalyzerException(
                    f"Category unknown for objective : '{priority}'")
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

        gpu_rows = []
        non_gpu_rows = []
        for measurement in measurements:
            for gpu_row in measurement.gpu_data().values():
                gpu_rows.append(gpu_row)
            non_gpu_rows.append(measurement.non_gpu_data())

        return Measurement(gpu_measurement=aggregation_func(gpu_rows),
                           non_gpu_measurement=aggregation_func(non_gpu_rows),
                           perf_config=None,
                           comparator=self)
