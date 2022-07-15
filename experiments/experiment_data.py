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

import re


class ExperimentData:
    """ 
    Class to hold and organize measurements for run configs
    """

    def __init__(self):
        self._data = {}

        self._best_run_config = None
        self._best_run_config_measurement = None
        self._missing_measurement_count = 0

    def add_run_config_measurement(self, run_config, run_config_measurement):
        """
        Add a run_config_measurement for the given run_config
        """
        if not run_config_measurement:
            return

        ma_key, pa_key = self._extract_run_config_keys(run_config)
        self._add_run_config_measurement_from_keys(ma_key, pa_key, run_config,
                                                   run_config_measurement)

    def get_run_config_measurement(self, run_config):
        """ 
        Get the run_config_measurement belonging to a given run_config
        """
        ma_key, pa_key = self._extract_run_config_keys(run_config)
        return self._get_run_config_measurement_from_keys(ma_key, pa_key)

    def get_model_config_count(self):
        """ 
        Get the total number of model configs in the data
        """
        return len(self._data.keys())

    def get_run_config_measurement_count(self):
        """ 
        Get the total number of measurements in the data
        """
        count = 0
        for ma_key in self._data.keys():
            count += len(self._data[ma_key].keys())
        return count

    def get_missing_measurement_count(self):
        return self._missing_measurement_count

    def get_best_run_config_measurement(self):
        """
        Get the best overall measurement in the data
        """
        return self._best_run_config_measurement

    def get_best_run_config(self):
        """
        Get the run_config corresponding to the best overall 
        run_config_measurement in the data
        """
        return self._best_run_config

    def _add_run_config_measurement_from_keys(self, ma_key, pa_key, run_config,
                                              run_config_measurement):
        self._update_best_trackers(run_config, run_config_measurement)

        curr_dict = self._data

        if ma_key not in curr_dict:
            curr_dict[ma_key] = {}
        curr_dict[ma_key][pa_key] = run_config_measurement

    def _update_best_trackers(self, run_config, run_config_measurement):
        if not self._best_run_config_measurement or run_config_measurement.get_non_gpu_metric_value(
                'perf_throughput'
        ) > self._best_run_config_measurement.get_non_gpu_metric_value(
                'perf_throughput'):
            self._best_run_config_measurement = run_config_measurement
            self._best_run_config = run_config

    def _get_run_config_measurement_from_keys(self, ma_key, pa_key):
        if ma_key not in self._data:
            print(f"WARNING: Model config {ma_key} not in results")
            self._missing_measurement_count += 1
            return None
        if pa_key not in self._data[ma_key]:
            print(
                f"WARNING: Model config {ma_key}, concurrency={pa_key} not in results"
            )
            self._missing_measurement_count += 1
            return None

        return self._data[ma_key][pa_key]

    def _extract_run_config_keys(self, run_config):

        model_config_key = ";".join([
            self._extract_model_config_key(x.model_config())
            for x in run_config.model_run_configs()
        ])
        perf_analyzer_key = ";".join([
            self._extract_perf_config_key(x.perf_config())
            for x in run_config.model_run_configs()
        ])

        return (model_config_key, perf_analyzer_key)

    def _extract_model_config_key(self, model_config):
        model_config_dict = model_config.get_config()
        max_batch_size = model_config_dict.get('max_batch_size', 0)
        instance_group = model_config_dict.get('instance_group', [{}])
        instance_count = instance_group[0].get('count', 0)
        key = f"instance_count={instance_count},max_batch_size={max_batch_size}"
        return key

    def _extract_perf_config_key(self, perf_config):
        pa_string = perf_config.to_cli_string()
        return self._make_pa_key_from_cli_string(pa_string)

    def _make_pa_key_from_cli_string(self, pa_cli_string):
        concurrencies = re.findall('--concurrency-range=(\d+)', pa_cli_string)
        batch_sizes = re.findall(' -b (\d+)', pa_cli_string)

        if len(concurrencies) != len(batch_sizes):
            raise Exception(f"concurrencies don't match batch sizes")

        for i in range(len(concurrencies)):
            tmp_int = int(concurrencies[i]) * int(batch_sizes[i])
            clamped_int = self._clamp_to_power_of_two(tmp_int)
            concurrencies[i] = str(clamped_int)

        return ';'.join(concurrencies)

    def _clamp_to_power_of_two(self, num):
        """ 
        Return the smallest power of two that is >= the input
        """
        v = 1
        while v < num:
            v *= 2
        return v
