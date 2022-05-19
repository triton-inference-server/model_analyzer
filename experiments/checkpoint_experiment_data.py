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

from experiments.experiment_data import ExperimentData
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from unittest.mock import MagicMock


class CheckpointExperimentData(ExperimentData):
    """
    Extends ExperimentData to be able to preload data from a checkpoint
    """

    def __init__(self, config):
        super().__init__()
        self._load_checkpoint(config)

    def _load_checkpoint(self, config):
        state_manager = AnalyzerStateManager(config, MagicMock())
        state_manager.load_checkpoint(checkpoint_required=True)

        results = state_manager.get_state_variable('ResultManager.results')
        # TODO: Multi-model
        model_name = config.profile_models[0].model_name()
        model_measurements = results.get_model_measurements_dict(model_name)
        for (run_config,
             run_config_measurements) in model_measurements.values():

            # Due to the way that data is stored in the AnalyzerStateManager, the
            # run_config only represents the model configuration used. The
            # perf_analyzer information for each measurement associated with it
            # is contained as a string in the run_config_measurements dict
            #
            (ma_key, _) = self._extract_run_config_keys(run_config)

            for (perf_analyzer_string,
                 run_config_measurement) in run_config_measurements.items():

                pa_key = self._make_pa_key_from_cli_string(perf_analyzer_string)
                self._add_run_config_measurement_from_keys(
                    [ma_key, pa_key], run_config, run_config_measurement)
