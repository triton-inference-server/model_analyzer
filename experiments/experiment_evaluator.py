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

from experiment_data import ExperimentData
from experiments.config_command_experiment import ConfigCommandExperiment


class ExperimentEvaluator:
    """ 
    Class to compare the results of a config generator execution against
    a checkpoint of raw data
    """

    def __init__(self, raw_data: ExperimentData, profile_data: ExperimentData,
                 config_command: ConfigCommandExperiment):
        self._raw_data = raw_data
        self._profile_data = profile_data

        self._maximize_throughput = True
        if config_command.min_throughput is not None:
            self._maximize_throughput = False

    def print_results(self):
        overall_best_measurement = self._raw_data.get_best_run_config_measurement(
        )
        overall_best_run_config = self._raw_data.get_best_run_config()
        generator_best_measurement = self._profile_data.get_best_run_config_measurement(
        )
        generator_best_run_config = self._profile_data.get_best_run_config()

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
        print(
            f"Overall best latency: {overall_best_measurement.get_non_gpu_metric_value('perf_latency_p99')}"
        )
        print()
        print(
            f"Generator num measurements: {self._profile_data.get_run_config_measurement_count()}"
        )
        print(
            f"Generator missing num measurements: {self._raw_data.get_missing_measurement_count()}"
        )
        print(
            f"Generator num configs: {self._profile_data.get_model_config_count()}"
        )
        print(
            f"Generator best config: {self._run_config_to_string(generator_best_run_config)}"
        )

        if generator_best_measurement:
            best_throughput = generator_best_measurement.get_non_gpu_metric_value(
                'perf_throughput')
            best_latency = generator_best_measurement.get_non_gpu_metric_value(
                'perf_latency_p99')
            overall_best_throughput = overall_best_measurement.get_non_gpu_metric_value(
                'perf_throughput')
            overall_best_latency = overall_best_measurement.get_non_gpu_metric_value(
                'perf_latency_p99')

            if self._maximize_throughput:
                percentile = round(best_throughput / overall_best_throughput, 2)
            else:
                percentile = round(overall_best_latency / best_latency, 2)
        else:
            best_throughput = None
            best_latency = None
            percentile = None

        print(f"Generator best throughput: {best_throughput}")
        print(f"Generator best latency: {best_latency}")
        print(f"Percentile: {percentile}")
        print()

    def _run_config_to_string(self, run_config):
        if run_config:
            str = "\n".join([
                f"{x.model_config().get_config()}"
                for x in run_config.model_run_configs()
            ])
            return str
        else:
            return None
