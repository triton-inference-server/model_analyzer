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

from profile_data import ProfileData

from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from unittest.mock import MagicMock
import re


class CheckpointData(ProfileData):
    """
    Extends ProfileData to be able to preload data from a checkpoint
    """

    def __init__(self, config):
        super().__init__()
        self._load_checkpoint(config)

    def _load_checkpoint(self, config):
        state_manager = AnalyzerStateManager(config, MagicMock())
        state_manager.load_checkpoint(checkpoint_required=True)

        results = state_manager.get_state_variable('ResultManager.results')
        model_name = config.profile_models[0].model_name()
        model_measurements = results.get_model_measurements_dict(model_name)
        for (run_config,
             run_config_measurements) in model_measurements.values():
            (ma_key, pa_key) = self._extract_run_config_keys(run_config)

            for (perf_analyzer_string,
                 measurement) in run_config_measurements.items():
                pa_key = self._make_pa_key_from_cli_string(perf_analyzer_string)
                self._add_measurement_from_keys([ma_key, pa_key], measurement)
