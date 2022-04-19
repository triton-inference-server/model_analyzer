# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .base_model_config_generator import BaseModelConfigGenerator

from model_analyzer.constants import LOGGER_NAME
import logging

logger = logging.getLogger(LOGGER_NAME)


class AutomaticModelConfigGenerator(BaseModelConfigGenerator):
    """ Given a model, generates model configs in automatic search mode """

    def __init__(self, config, model, client, default_only):
        """
        Parameters
        ----------
        config: ModelAnalyzerConfig
        model: The model to generate ModelConfigs for
        client: TritonClient
        default_only: Bool 
            If true, only the default config will be generated
            If false, the default config will NOT be generated
        """
        super().__init__(config, model, client, default_only)

        self._max_instance_count = config.run_config_search_max_instance_count
        self._min_instance_count = config.run_config_search_min_instance_count
        self._max_model_batch_size = config.run_config_search_max_model_batch_size
        self._min_model_batch_size = config.run_config_search_min_model_batch_size

        self._instance_kind = "KIND_CPU" if self._cpu_only else "KIND_GPU"

        self._curr_instance_count = self._min_instance_count
        self._curr_max_batch_size = 0

        self._sweep_max_batch_size_disabled = self._determine_sweep_max_batch_size_disabled(
        )
        self._max_batch_size_warning_printed = False

        # Contains the max throughput from each provided list of measurements
        # since the last time we stepped max_batch_size
        #
        self._curr_max_batch_size_throughputs = []

        self._reset_max_batch_size()

    def _done_walking(self):
        return self._done_walking_max_batch_size() \
           and self._done_walking_instance_count()

    def _step(self):
        if self._done_walking_max_batch_size():
            self._reset_max_batch_size()
            self._step_instance_count()
        else:
            self._step_max_batch_size()

    def _step_max_batch_size(self):
        self._curr_max_batch_size *= 2

        last_max_throughput = self._get_last_results_max_throughput()
        self._curr_max_batch_size_throughputs.append(last_max_throughput)

    def _step_instance_count(self):
        self._curr_instance_count += 1

    def _done_walking_max_batch_size(self):
        if self._max_batch_size_limit_reached():
            return True

        if not self._last_results_increased_throughput():
            if not self._max_batch_size_warning_printed:
                logger.info(
                    "No longer increasing max_batch_size because throughput has plateaued"
                )
                self._max_batch_size_warning_printed = True
            return True
        return False

    def _done_walking_instance_count(self):
        return self._curr_instance_count == self._max_instance_count

    def _max_batch_size_limit_reached(self):
        return (self._curr_max_batch_size * 2) > self._max_model_batch_size

    def _last_results_erroneous(self):
        for measurement in self._last_results:
            if measurement is None:
                return True
        return False

    def _last_results_increased_throughput(self):
        max_throughput = self._get_last_results_max_throughput()
        max_throughput_increased = all(
            max_throughput is not None and max_throughput > t
            for t in self._curr_max_batch_size_throughputs)

        return max_throughput_increased

    def _reset_max_batch_size(self):
        if self._sweep_max_batch_size_disabled:
            self._curr_max_batch_size = self._max_model_batch_size
        else:
            self._curr_max_batch_size = self._min_model_batch_size
        self._curr_max_batch_size_throughputs = []
        self._max_batch_size_warning_printed = False

    def _get_last_results_max_throughput(self):
        throughputs = [
            m.get_metric_value('perf_throughput')
            for m in self._last_results
            if m is not None
        ]
        if not throughputs:
            return None
        else:
            return max(throughputs)

    def _get_next_model_config(self):
        param_combo = self._get_curr_param_combo()
        model_config = self._make_direct_mode_model_config(param_combo)
        return model_config

    def _get_curr_param_combo(self):
        if self._default_only:
            return self.DEFAULT_PARAM_COMBO

        config = {
            'dynamic_batching': {},
            'instance_group': [{
                'count': self._curr_instance_count,
                'kind': self._instance_kind
            }]
        }

        if not self._sweep_max_batch_size_disabled:
            config['max_batch_size'] = self._curr_max_batch_size

        return config

    def _determine_sweep_max_batch_size_disabled(self):
        config = self._get_base_model_config_dict()
        max_batch_size_disabled = False
        if "max_batch_size" not in config or config['max_batch_size'] == 0:
            max_batch_size_disabled = True
        return max_batch_size_disabled
