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

from model_analyzer.result.results import Results
from model_analyzer.record.record import RecordType


class AnalyzerState:
    """
    All the state information needed by 
    Model Analyzer in one place
    """

    def __init__(self):
        self._state_dict = {}

    def to_dict(self):
        return self._state_dict

    @classmethod
    def from_dict(cls, state_dict):
        state = AnalyzerState()

        # Model Variant mapping
        state._state_dict[
            'ModelManager.model_variant_name_manager'] = state_dict[
                'ModelManager.model_variant_name_manager']

        # Fill results
        state._state_dict['ResultManager.results'] = Results.from_dict(
            state_dict['ResultManager.results'])

        # Server data
        state._state_dict['ResultManager.server_only_data'] = {}
        for gpu_uuid, gpu_data_list in state_dict[
                'ResultManager.server_only_data'].items():
            metric_list = []
            for [tag, record_dict] in gpu_data_list:
                record_type = RecordType.get(tag)
                record = record_type.from_dict(record_dict)
                metric_list.append(record)
            state._state_dict['ResultManager.server_only_data'][
                gpu_uuid] = metric_list

        # GPU data
        state._state_dict['MetricsManager.gpus'] = state_dict[
            'MetricsManager.gpus']

        return state

    def get(self, name):
        if name in self._state_dict:
            return self._state_dict[name]
        else:
            return None

    def set(self, name, value):
        self._state_dict[name] = value
