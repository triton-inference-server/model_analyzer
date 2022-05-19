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

    def add_run_config_measurement(self, run_config, run_config_measurement):
        """
        Add a run_config_measurement for the given run_config
        """
        keys = self._extract_run_config_keys(run_config)
        self._add_run_config_measurement_from_keys(keys, run_config,
                                                   run_config_measurement)

    def get_run_config_measurement(self, run_config):
        """ 
        Get the run_config_measurement belonging to a given run_config
        """
        keys = self._extract_run_config_keys(run_config)
        return self._get_run_config_measurement_from_keys(keys)

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

    def _add_run_config_measurement_from_keys(self, keys, run_config,
                                              run_config_measurement):
        self._update_best_trackers(run_config, run_config_measurement)

        curr_dict = self._data

        ma_key, pa_key = keys

        if ma_key not in curr_dict:
            curr_dict[ma_key] = {}
        curr_dict[ma_key][pa_key] = run_config_measurement

    def _update_best_trackers(self, run_config, run_config_measurement):
        if not self._best_run_config_measurement or run_config_measurement > self._best_run_config_measurement:
            self._best_run_config_measurement = run_config_measurement
            self._best_run_config = run_config

    def _get_run_config_measurement_from_keys(self, keys):
        ma_key, pa_key = keys

        if ma_key not in self._data:
            raise Exception(f"Model config {ma_key} not in results")
        if pa_key not in self._data[ma_key]:
            raise Exception(f"PA config {pa_key} not in results for {ma_key}")

        return self._data[ma_key][pa_key]

    def _extract_run_config_keys(self, run_config):

        # TODO: need to update the keys for multi-model
        model_config = run_config.model_run_configs()[0].model_config()
        perf_config = run_config.model_run_configs()[0].perf_config()

        model_config_key = self._extract_model_config_key(model_config)
        perf_analyzer_key = self._extract_perf_config_key(perf_config)

        return (model_config_key, perf_analyzer_key)

    def _extract_model_config_key(self, model_config):
        model_config_dict = model_config.get_config()
        max_batch_size = model_config_dict.get('max_batch_size', 0)
        instance_group = model_config_dict.get('instance_group', [{}])
        instance_count = instance_group[0].get('count', 0)
        key = f"max_batch_size={max_batch_size},instance_count={instance_count}"
        return key

    def _extract_perf_config_key(self, perf_config):
        pa_string = perf_config.to_cli_string()
        return self._make_pa_key_from_cli_string(pa_string)

    def _make_pa_key_from_cli_string(self, pa_cli_string):
        concurrency_group = re.search('--concurrency-range=(\d+)',
                                      pa_cli_string)
        concurrency = int(concurrency_group.group(1))
        key = f"concurrency={concurrency}"
        return key
