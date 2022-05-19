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


class ProfileData:
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
        self._add_run_config_measurement_from_keys(keys, run_config_measurement)

    def get_run_config_measurement(self, run_config):
        """ 
        Get the run_config_measurement belonging to a given run_config
        """
        keys = self._extract_run_config_keys(run_config)
        return self._get_run_config_measurement_from_keys(keys)

    def get_config_count(self):
        """ 
        Get the total number of model configs in the data
        """
        count = 0
        for ma_key in self._data.keys():
            count += 1
        return count

    def get_run_config_measurement_count(self):
        """ 
        Get the total number of measurements in the data
        """
        count = 0
        for ma_key in self._data.keys():
            for pa_key in self._data[ma_key].keys():
                count += 1
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

    def _add_run_config_measurement_from_keys(self, keys,
                                              run_config_measurement):
        self._update_best_trackers(keys, run_config_measurement)

        curr_dict = self._data

        for i in range(len(keys) - 1):
            if keys[i] not in curr_dict:
                curr_dict[keys[i]] = {}
            curr_dict = curr_dict[keys[i]]
        curr_dict[keys[-1]] = run_config_measurement

    def _update_best_trackers(self, keys, run_config_measurement):
        if not self._best_run_config_measurement or run_config_measurement > self._best_run_config_measurement:
            self._best_run_config_measurement = run_config_measurement
            self._best_run_config = keys[0]

    def _get_run_config_measurement_from_keys(self, keys):
        curr_dict = self._data

        for i in range(len(keys) - 1):
            if keys[i] not in curr_dict:
                raise Exception(f"keys {keys} not in results")
            curr_dict = curr_dict[keys[i]]

        if keys[-1] not in curr_dict:
            raise Exception(f"final key of keys {keys} not in results")
        return curr_dict[keys[-1]]

    def _extract_run_config_keys(self, run_config):

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
