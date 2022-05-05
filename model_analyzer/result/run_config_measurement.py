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

from model_analyzer.constants import COMPARISON_SCORE_THRESHOLD
from model_analyzer.constants import LOGGER_NAME

from model_analyzer.result.model_config_measurement import ModelConfigMeasurement
from model_analyzer.record.record import RecordType

from statistics import mean

from functools import total_ordering
import logging

logger = logging.getLogger(LOGGER_NAME)


@total_ordering
class RunConfigMeasurement:
    """
    Encapsulates the set of metrics obtained from all model configs
    in a single RunConfig
    """

    def __init__(self, model_variants_name, gpu_data):
        """
        model_variants_name: str
            Name of the model variants this measurement was collected for
        
        gpu_data : dict of list of Records
            Metrics from the monitors that have a GPU UUID
            associated with them            
        """
        self._model_variants_name = model_variants_name

        self._gpu_data = gpu_data
        self._avg_gpu_data = self._average_list(list(self._gpu_data.values()))
        self._avg_gpu_data_from_tag = self._get_avg_gpu_data_from_tag()

        self._model_config_measurements = []
        self._model_config_weights = []

    @classmethod
    def from_dict(cls, run_config_measurement_dict):
        run_config_measurement = RunConfigMeasurement(None, {})

        run_config_measurement._model_variants_name = run_config_measurement_dict[
            '_model_variants_name']

        run_config_measurement._gpu_data = cls._deserialize_gpu_data(
            run_config_measurement, run_config_measurement_dict['_gpu_data'])

        run_config_measurement._avg_gpu_data = cls._average_list(
            run_config_measurement,
            list(run_config_measurement._gpu_data.values()))

        run_config_measurement._avg_gpu_data_from_tag = cls._get_avg_gpu_data_from_tag(
            run_config_measurement)

        run_config_measurement._model_config_measurements = cls._deserialize_model_config_measurements(
            run_config_measurement,
            run_config_measurement_dict['_model_config_measurements'])

        run_config_measurement._model_config_weights = run_config_measurement_dict[
            '_model_config_weights']

        return run_config_measurement

    def set_model_config_weighting(self, model_config_weights):
        """
        Sets the model config weightings used when calculating 
        weighted metrics
        
        Parameters
        ----------
        model_weights: list of ints
        Weights are the relative importance of the model_configs 
        with respect to one another
        """
        self._model_config_weights = [
            model_config_weight / sum(model_config_weights)
            for model_config_weight in model_config_weights
        ]

    def add_model_config_measurement(self, model_config_name,
                                     model_specific_pa_params, non_gpu_data):
        """ 
        Adds a measurement from a single model config in this PA's run 
        
        model_config_name : string
            The model config name that was used for this PA run
        model_specific_pa_params: dict
            Dictionary of PA parameters that change between models
            in a multi-model run
        non_gpu_data : list of Records
            Metrics that do not have a GPU UUID associated with them,
            from either CPU or PA
        """
        self._model_config_measurements.append(
            ModelConfigMeasurement(model_config_name, model_specific_pa_params,
                                   non_gpu_data))

        # By default setting all models to have equal weighting
        self._model_config_weights.append(1)

    def set_metric_weightings(self, metric_objectives):
        """
        Sets the metric weighting for all non-GPU measurements
        
        Parameters
        ----------
        metric_objectives : list of dict of RecordTypes
            One entry per ModelConfig
        """
        for index, measurement in enumerate(self._model_config_measurements):
            measurement.set_metric_weighting(metric_objectives[index])

    def model_variants_name(self):
        """
        Returns: str
            The name of the model variants this measurement was collected for
        """

        return self._model_variants_name

    def model_name(self):
        """
        Returns
        -------
        str: Model name for this RunConfigMeasurement
        """

        return self._model_name

    def data(self):
        """
        Returns
        -------
        per model list of a list of Records
            All the metric values in each model's measurement 
            for both GPU and non-GPU
        """

        return [
            self._avg_gpu_data + mcm.non_gpu_data()
            for mcm in self._model_config_measurements
        ]

    def gpu_data(self):
        """
        Returns
        -------
        List of Records
            GPU specific measurements
        """

        return self._gpu_data

    def non_gpu_data(self):
        """
        Returns
        -------
        per model list of a list Records
            The non GPU specific measurements
        """

        return [
            model_config_measurement.non_gpu_data()
            for model_config_measurement in self._model_config_measurements
        ]

    def get_gpu_metric(self, tag):
        """
        Returns the average of Records associated with this GPU metric
        
        Parameters
        ----------
        tag : str
            A human readable tag that corresponds
            to a particular GPU metric

        Returns
        -------
        Record:
            of average GPU metric Records corresponding to this tag,
            or None if tag not found
        """
        if tag in self._avg_gpu_data_from_tag:
            return self._avg_gpu_data_from_tag[tag]
        else:
            logger.warning(
                f"No GPU metric corresponding to tag '{tag}' "
                "found in the model's measurement. Possibly comparing "
                "measurements across devices.")
            return None

    def get_non_gpu_metric(self, tag):
        """
        Returns the Records associated with this non-GPU metric
        
        Parameters
        ----------
        tag : str
            A human readable tag that corresponds
            to a particular metric

        Returns
        -------
        list:
            of per model list:
                of non-GPU metric Records, or None if tag not found
        """
        return [
            model_config_measurement.get_metric(tag)
            for model_config_measurement in self._model_config_measurements
        ]

    def get_weighted_non_gpu_metric(self, tag):
        """
        Parameters
        ----------
        tag : str
            A human readable tag that corresponds
            to a particular non-GPU metric

        Returns
        -------
        list:
            of per model list:
                of weighted non-GPU metric Records, 
                or None if tag not found
        
        """
        assert len(self._model_config_weights) == len(
            self._model_config_measurements)

        return [
            model_config_measurement.get_metric(tag) *
            self._model_config_weights[index]
            for index, model_config_measurement in enumerate(
                self._model_config_measurements)
        ]

    def get_non_gpu_metric_value(self, tag, default_value=0):
        """
        Parameters
        ----------
        tag : str
            A human readable tag that corresponds
            to a particular non-GPU metric
        default_value : any
            Value to return if tag is not found

        Returns
        -------
        Float
            Compuation of the values of the non-GPU metric Records 
            corresponding to the tag, default_value if tag not found,
            based on the supplied aggregation function (usually mean or sum).
        """
        return RecordType.get_all_record_types()[tag].value_function()([
            default_value if m is None else m.value()
            for m in self.get_non_gpu_metric(tag)
        ])

    def get_gpu_metric_value(self, tag, default_value=0):
        """
        Parameters
        ----------
        tag : str
            A human readable tag that corresponds
            to a particular GPU metric
        default_value : any
            Value to return if tag is not found

        Returns
        -------
        Record : 
            Average of the values of the GPU metric Records 
            corresponding to the tag, default_value if tag not found.
        """
        return default_value if self.get_gpu_metric(
            tag) is None else self.get_gpu_metric(tag).value()

    def get_weighted_non_gpu_metric_value(self, tag, default_value=0):
        """
        Parameters
        ----------
        tag : str
            A human readable tag that corresponds
            to a particular metric
        default_value : any
            Value to return if tag is not found

        Returns
        -------
        list of Records
            Weighted average of the values of the metric Record corresponding 
            to the tag, default_value if tag not found.
        """
        assert len(self._model_config_weights) == len(
            self._model_config_measurements)

        weighted_non_gpu_metrics = [
            metric.value() * self._model_config_weights[index]
            for index, metric in enumerate(self.get_non_gpu_metric(tag))
        ]

        return RecordType.get_all_record_types()[tag].value_function()(
            weighted_non_gpu_metrics)

    def gpus_used(self):
        """
        Returns
        -------
        list of ints
            list of device IDs used in this measurement
        """

        return list(self._gpu_data.keys())

    def model_specific_pa_params(self):
        """
        Returns
        -------
        list:
            of dicts:
                of model specific PA parameters
                used in this measurement
        """

        return [
            model_config_measurement.model_specific_pa_params()
            for model_config_measurement in self._model_config_measurements
        ]

    def is_better_than(self, other):
        """
        Checks whether a measurement is better than another
        by using the weighted average across all model configs in the
        RunConfig
        
        If True, this means this RunConfig measurement is better 
        than the other
        """
        # seems like this should be == -1 but we're using a min heap
        return self._compare_measurements(other) == 1

    def __eq__(self, other):
        """
        Check whether two sets of measurements are equivalent
        """

        return self._compare_measurements(other) == 0

    def __lt__(self, other):
        """
        Checks whether a measurement is better than another
        by using the weighted average across all model configs in the
        RunConfig
        
        This is used when sorting
        
        Returns
        -------
        bool:
            True if other is better than or equal to self
        """

        return not self.is_better_than(other)

    def _compare_measurements(self, other):
        """
        Compares two RunConfig measurements based on each
        ModelConfigs weighted metric objectives and the 
        ModelConfigs weighted value within the RunConfig
        
        Parameters
        ----------
        other: RunConfigMeasurement

        Returns
        -------
        int
            0
                if the results are determined
                to be the same within a threshold
            1
                if self > other (is better than)
            -1
                if self < other (is worse than)
        """

        # Step 1: for each ModelConfig determine the weighted score
        weighted_mcm_scores = self._calculate_weighted_mcm_score(other)

        # Step 2: combine these using the ModelConfig weighting
        weighted_rcm_score = self._calculate_weighted_rcm_score(
            weighted_mcm_scores)

        # Step 3: check the polarity
        if weighted_rcm_score > COMPARISON_SCORE_THRESHOLD:
            return 1
        elif weighted_rcm_score < -COMPARISON_SCORE_THRESHOLD:
            return -1
        return 0

    def _calculate_weighted_mcm_score(self, other):
        """
        Parameters
        ----------
        other: RunConfigMeasurement

        Returns
        -------
        list of floats
            A weighted score for each ModelConfig measurement in the RunConfig
        """
        return [
            self_mcm.get_weighted_score(other_mcm)
            for self_mcm, other_mcm in zip(self._model_config_measurements,
                                           other._model_config_measurements)
        ]

    def _calculate_weighted_rcm_score(self, weighted_mcm_scores):
        """
        Parameters
        ----------
        weighted_mcm_scores: list of floats
            A weighted score for each ModelConfig measurement in the RunConfig  

        Returns
        -------
        float
            The weighted score. A positive value indicates this 
            RunConfig measurement is better than the other
        """
        assert len(self._model_config_weights) == len(weighted_mcm_scores)

        return sum([
            weighted_mcm_score * model_config_weight
            for weighted_mcm_score, model_config_weight in zip(
                weighted_mcm_scores, self._model_config_weights)
        ])

    def _average_list(self, row_list):
        """
        Average a 2d list
        """

        if not row_list:
            return row_list
        else:
            N = len(row_list)
            d = len(row_list[0])
            avg = [0 for _ in range(d)]
            for i in range(d):
                avg[i] = (sum([row_list[j][i] for j in range(1, N)],
                              start=row_list[0][i]) * 1.0) / N
            return avg

    def _deserialize_gpu_data(self, serialized_gpu_data):
        gpu_data = {}
        for gpu_uuid, gpu_data_list in serialized_gpu_data.items():
            metric_list = []
            for [tag, record_dict] in gpu_data_list:
                record_type = RecordType.get(tag)
                record = record_type.from_dict(record_dict)
                metric_list.append(record)
            gpu_data[gpu_uuid] = metric_list

        return gpu_data

    def _get_avg_gpu_data_from_tag(self):
        return {type(metric).tag: metric for metric in self._avg_gpu_data}

    def _deserialize_model_config_measurements(
            self, serialized_model_config_measurements):
        model_config_measurements = []
        for mcm_dict in serialized_model_config_measurements:
            model_config_measurements.append(
                ModelConfigMeasurement.from_dict(mcm_dict))

        return model_config_measurements
