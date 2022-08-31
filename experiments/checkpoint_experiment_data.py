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
from copy import deepcopy
import re


class CheckpointExperimentData(ExperimentData):
    """
    Extends ExperimentData to be able to preload data from a checkpoint
    """

    LOAD_ONLY_VISABLE = True

    def __init__(self, config):
        super().__init__()
        self._default_run_config = None
        self._load_checkpoint(config)

    def get_default_config_dict(self):
        ret = self._default_run_config.model_run_configs()[0].model_config(
        ).to_dict()
        ret = deepcopy(ret)
        del ret["cpu_only"]
        return ret

    def _load_checkpoint(self, config):
        state_manager = AnalyzerStateManager(config, MagicMock())
        state_manager.load_checkpoint(checkpoint_required=True)

        results = state_manager.get_state_variable('ResultManager.results')

        model_name = ",".join([x.model_name() for x in config.profile_models])
        model_measurements = results.get_model_measurements_dict(model_name)
        for (run_config,
             run_config_measurements) in model_measurements.values():

            if run_config.model_variants_name(
            ) == model_name + "_config_default":
                self._default_run_config = run_config

            # Due to the way that data is stored in the AnalyzerStateManager, the
            # run_config only represents the model configuration used. The
            # perf_analyzer information for each measurement associated with it
            # is contained as a string in the run_config_measurements dict
            #
            ma_key, _ = self._extract_run_config_keys(run_config)

            for (perf_analyzer_string,
                 run_config_measurement) in run_config_measurements.items():

                run_config_measurement.set_model_config_constraints(
                    model_config_constraints=[config.constraints])
                run_config_measurement.set_metric_weightings(
                    metric_objectives=[config.objectives])
                pa_key = self._make_pa_key_from_cli_string(perf_analyzer_string)

                if CheckpointExperimentData.LOAD_ONLY_VISABLE:
                    if not self._are_keys_visable_to_algorithm(ma_key, pa_key):
                        continue

                existing_measurement = self._get_run_config_measurement_from_keys(
                    ma_key, pa_key, skip_warn=True)
                if not existing_measurement or run_config_measurement > existing_measurement:
                    self._add_run_config_measurement_from_keys(
                        ma_key, pa_key, run_config, run_config_measurement)

        if self._default_run_config is None:
            print(f"No default config for {model_name}")
            exit(1)

        self._print_map()

    def _print_map(self):
        row_str = "\t\t"
        for j in range(0, 10):
            row_str += f"    [{j}]\t\t"
        print(row_str)

        has_exponential_inst_count = self._has_exponential_inst_count()

        for i in range(0, 100):
            row_has_data = False
            row_str = f"\t[{i}]"
            for j in range(0, 10):
                instance_count = 2**j if has_exponential_inst_count else j + 1
                max_batch_size = 2**i

                ma_key = f"instance_count={instance_count},max_batch_size={max_batch_size}"

                clamped_int = self._clamp_to_power_of_two(2 * instance_count *
                                                          max_batch_size)

                pa_key = str(clamped_int)

                measurement = self._get_run_config_measurement_from_keys(
                    ma_key, pa_key, skip_warn=True)
                tput = 0
                lat = 0
                if measurement:
                    row_has_data = True
                    tput = measurement.get_non_gpu_metric_value(
                        'perf_throughput')
                    lat = measurement.get_non_gpu_metric_value(
                        'perf_latency_p99')
                row_str += f"\t{tput:4.1f}:{lat:4.1f}"

            # Print at least the first 7 rows, and then stop when we hit an empty row
            # (Some databases don't have data for the first 6 rows)
            if row_has_data or i < 7:
                print(row_str)
            else:
                break

    def _has_exponential_inst_count(self) -> bool:
        # See if instance count of 3 is in the database. If not, it is exponential (1,2,4,8)
        ret = False
        ma_key = f"instance_count=3,max_batch_size=1"
        pa_key = "8"
        measurement = self._get_run_config_measurement_from_keys(ma_key,
                                                                 pa_key,
                                                                 skip_warn=True)
        if not measurement:
            ret = True
        return ret

    def _are_keys_visable_to_algorithm(self, ma_key, pa_key):
        # The quick algorithm can only see meaurements where the
        # concurrency is 2 * inst_count * max_batch_size.
        results = re.search("instance_count=(\d+),max_batch_size=(\d+)", ma_key)
        inst_count = int(results.group(1))
        max_batch_size = int(results.group(2))
        expected_pa_key = self._clamp_to_power_of_two(2 * inst_count *
                                                      max_batch_size)
        return expected_pa_key == int(pa_key)
