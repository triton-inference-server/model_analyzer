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


class ResultEvaluator:
    """ 
    Class to compare the results of a config generator execution against
    a checkpoint of raw data
    """

    def __init__(self, raw_data, profile_data):
        self._raw_data = raw_data
        self._profile_data = profile_data

    def print_results(self):
        overall_best_measurement = self._raw_data.get_best_run_config_measurement(
        )
        overall_best_run_config = self._raw_data.get_best_run_config()
        generator_best_measurement = self._profile_data.get_best_run_config_measurement(
        )
        generator_best_run_config = self._raw_data.get_best_run_config()

        print()
        print("====================================")
        print(
            f"Overall num measurements: {self._raw_data.get_run_config_measurement_count()}"
        )
        print(f"Overall num configs: {self._raw_data.get_model_config_count()}")
        print(
            f"Overall best config: {self._run_config_to_string(overall_best_run_config)}"
        )
        print(
            f"Overall best throughput: {overall_best_measurement.get_non_gpu_metric_value('perf_throughput')}"
        )
        print()
        print(
            f"Generator num measurements: {self._profile_data.get_run_config_measurement_count()}"
        )
        print(
            f"Generator num configs: {self._profile_data.get_model_config_count()}"
        )
        print(
            f"Generator best config: {self._run_config_to_string(generator_best_run_config)}"
        )
        print(
            f"Generator best throughput: {generator_best_measurement.get_non_gpu_metric_value('perf_throughput')}"
        )
        print()

    def _run_config_to_string(self, run_config):
        # TODO: Multi-model
        mc = run_config.model_run_configs()[0].model_config()
        return f"{mc.get_config()}"
