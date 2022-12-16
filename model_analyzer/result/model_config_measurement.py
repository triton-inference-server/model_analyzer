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

from model_analyzer.record.record import RecordType

from copy import deepcopy
from statistics import mean
from functools import total_ordering
import logging

logger = logging.getLogger(LOGGER_NAME)


@total_ordering
class ModelConfigMeasurement:
    """
    Encapsulates the set of non-gpu metrics obtained from a single model config's
    RunConfig run
    """

    def __init__(self, model_config_name, model_specific_pa_params,
                 non_gpu_data):
        """
        model_config_name : string
            The model config name that was used in the RunConfig
        model_specific_pa_params: dict
            Dictionary of PA parameters that can change between models
            in a multi-model RunConfig
        non_gpu_data : list of Records
            Metrics that do not have a GPU UUID associated with them,
            from either CPU or PA
        """

        self._model_config_name = model_config_name
        self._model_specific_pa_params = model_specific_pa_params
        self._non_gpu_data = non_gpu_data

        self._non_gpu_data_from_tag = self._get_non_gpu_data_from_tag()

        # Set a default metric weighting
        self._metric_weights = {"perf_throughput": 1}

    def to_dict(self):
        mcm_dict = deepcopy(self.__dict__)
        del mcm_dict['_metric_weights']

        return mcm_dict

    @classmethod
    def from_dict(cls, model_config_measurement_dict):
        model_config_measurement = ModelConfigMeasurement(None, {}, [])

        model_config_measurement._model_config_name = model_config_measurement_dict[
            '_model_config_name']
        model_config_measurement._model_specific_pa_params = model_config_measurement_dict[
            '_model_specific_pa_params']

        model_config_measurement._non_gpu_data = cls._deserialize_non_gpu_data(
            model_config_measurement,
            model_config_measurement_dict['_non_gpu_data'])

        model_config_measurement._non_gpu_data_from_tag = cls._get_non_gpu_data_from_tag(
            model_config_measurement)

        return model_config_measurement

    def set_metric_weighting(self, metric_objectives):
        """
        Sets the metric weighting for this measurement based
        on the objectives
        
        Parameters
        ----------
        metric_objectives : dict of RecordTypes
            keys are the metric types, and values are the relative importance
            of the keys with respect to each other
        """
        self._metric_weights = {
            objective: (value / sum(metric_objectives.values()))
            for objective, value in metric_objectives.items()
        }

    def model_config_name(self):
        """
        Return the model_config name
        used to get this measurement
        """

        return self._model_config_name

    def model_name(self):
        """
        Return the model name

        TODO: This method should be replaced with the extract_model_name_from_variant_name() static method,
        once the ensemble code gets merged to main
        """

        return self._model_config_name.partition("_config_")[0]

    def model_specific_pa_params(self):
        """
        Return a dict of model specific PA parameters
        used in this measurement
        """

        return self._model_specific_pa_params

    def non_gpu_data(self):
        """
        Return a list of the non-GPU specific 
        measurement Records
        """

        return self._non_gpu_data

    def get_metric(self, tag):
        """
        Parameters
        ----------
        tag : str
            A human readable tag that corresponds
            to a particular metric

        Returns
        -------
        Record
            metric Record corresponding to
            the tag, in this measurement, None
            if tag not found.
        """

        if tag in self._non_gpu_data_from_tag:
            return self._non_gpu_data_from_tag[tag]
        else:
            return None

    def get_metric_value(self, tag, default_value=0):
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
        Record
            Value of the metric Record corresponding 
            to the tag, in this measurement, 
            default_value if tag not found.
        """

        metric = self.get_metric(tag)
        if metric is None:
            return default_value
        return metric.value()

    def get_weighted_score(self, other):
        """ 
        Parameters
        ----------
        other: ModelConfigMeasurement
            set of (non_gpu) metrics to be compared against
        
        Returns
        -------
        float
            The weighted score between this ModelConfig
            and the other ModelConfig
        """
        return self._calculate_weighted_score(other)

    def is_better_than(self, other):
        """
        Checks whether a measurement is better than
        another

        If True, this means this measurement is better
        than the other.
        
        Parameters
        ----------
        other: ModelConfigMeasurement
            set of (non_gpu) metrics to be compared against
        """

        return self._compare_measurements(other) == 1

    def __eq__(self, other):
        """
        Check whether two sets of measurements are equivalent
        
        Parameters
        ----------
        other: ModelConfigMeasurement
            set of (non_gpu) metrics to be compared against
        """

        return self._compare_measurements(other) == 0

    def __lt__(self, other):
        """
        Checks whether a measurement is better than
        another

        This is used when sorting
        
        Parameters
        ----------
        other: ModelConfigMeasurement
            set of (non_gpu) metrics to be compared against
            
        Returns
        -------
        bool:
            True if other is better than or equal to self
        """

        return not self.is_better_than(other)

    def _compare_measurements(self, other):
        """
        Compares two ModelConfig measurements 
        based on the weighted metric objectives

        Parameters
        ----------
        other : ModelConfigMeasurement
            set of (non_gpu) metrics to be compared against

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
        weighted_score = self._calculate_weighted_score(other)

        if weighted_score > COMPARISON_SCORE_THRESHOLD:
            return 1
        elif weighted_score < -COMPARISON_SCORE_THRESHOLD:
            return -1
        return 0

    def _calculate_weighted_score(self, other):
        """
        Calculates the weighted score between two 
        ModelConfig measurements based on the weighted
        metric objectives
        
        Parameters
        ----------
        other : ModelConfigMeasurement
            set of (non_gpu) metrics to be compared against
            
        Returns
        -------
        float
            The weighted score. A positive value indicates 
            this ModelConfig measurement is better than the other
        """

        weighted_score = 0.0
        for objective, weight in self._metric_weights.items():
            self_metric = self.get_metric(tag=objective)
            other_metric = other.get_metric(tag=objective)

            # Handle the case where objective GPU metric is queried on CPU only
            if self_metric and other_metric is None:
                return 1
            elif other_metric and self_metric is None:
                return -1
            elif self_metric is None and other_metric is None:
                return 0

            metric_diff = self_metric - other_metric
            average = mean([self_metric.value(), other_metric.value()])
            weighted_score += weight * (metric_diff.value() / average)

        return weighted_score

    def calculate_weighted_percentage_gain(self, other):
        """
        Calculates the weighted percentage between two 
        ModelConfig measurements based on the weighted
        metric objectives
        
        Parameters
        ----------
        other : ModelConfigMeasurement
            set of (non_gpu) metrics to be compared against
            
        Returns
        -------
        float
            The weighted percentage gain. A positive value indicates 
            this ModelConfig measurement is better than the other
        """

        weighted_pct = 0.0
        for objective, weight in self._metric_weights.items():
            self_metric = self.get_metric(tag=objective)
            other_metric = other.get_metric(tag=objective)

            # Handle the case where objective GPU metric is queried on CPU only
            if self_metric and other_metric is None:
                return 100
            elif other_metric and self_metric is None:
                return -100
            elif self_metric is None and other_metric is None:
                return 0

            metric_pct = self_metric.calculate_percentage_gain(other_metric)

            weighted_pct += metric_pct * weight

        return weighted_pct

    def _get_non_gpu_data_from_tag(self):
        return {metric.tag: metric for metric in self._non_gpu_data}

    def _deserialize_non_gpu_data(self, serialized_non_gpu_data):
        non_gpu_data = []

        for [tag, record_dict] in serialized_non_gpu_data:
            record_type = RecordType.get(tag)
            record = record_type.from_dict(record_dict)
            non_gpu_data.append(record)

        return non_gpu_data
