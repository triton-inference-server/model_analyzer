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

from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.result.measurement import Measurement
from model_analyzer.record.record import RecordType


class AnalyzerState:
    """
    All the state information needed by 
    Model Analyzer in one place
    """
    def __init__(self):
        self._state_dict = {}

    def serialize(self):
        return self._state_dict

    def deserialize(self, state_dict):
        # Fill results
        self._state_dict['ResultManager.results'] = {}
        for model_name in state_dict['ResultManager.results']:
            self._state_dict['ResultManager.results'][model_name] = {}
            for model_config_name in state_dict['ResultManager.results'][
                    model_name]:
                model_config_dict, measurements = state_dict[
                    'ResultManager.results'][model_name][model_config_name]

                # Deserialize model config
                model_config = ModelConfig(None)
                model_config.deserialize(model_config_dict)

                # Deserialize measurements
                measurements_dict = {}
                for measurement_key, measurement_dict in measurements.items():
                    measurement = Measurement({}, [], None)
                    measurement.deserialize(measurement_dict)
                    measurements_dict[measurement_key] = measurement
                self._state_dict['ResultManager.results'][model_name][
                    model_config_name] = (model_config, measurements_dict)

        # Server data
        self._state_dict['ResultManager.server_only_data'] = {}
        for gpu_uuid, gpu_data_list in state_dict[
                'ResultManager.server_only_data'].items():
            metric_list = []
            for [tag, record_dict] in gpu_data_list:
                record_type = RecordType.get(tag)
                record = record_type(0)
                record.deserialize(record_dict)
                metric_list.append(record)
            self._state_dict['ResultManager.server_only_data'][
                gpu_uuid] = metric_list

        # GPU data
        self._state_dict['MetricsManager.gpus'] = state_dict[
            'MetricsManager.gpus']

    def get(self, name):
        if name in self._state_dict:
            return self._state_dict[name]
        else:
            return None

    def set(self, name, value):
        self._state_dict[name] = value
