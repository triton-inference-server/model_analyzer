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

from .config_generator_interface import ConfigGeneratorInterface
from .generator_utils import GeneratorUtils as utils

from model_analyzer.constants import THROUGHPUT_MINIMUM_GAIN, THROUGHPUT_MINIMUM_CONSECUTIVE_TRIES
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig

from model_analyzer.constants import LOGGER_NAME
import logging

logger = logging.getLogger(LOGGER_NAME)


class PerfAnalyzerConfigGenerator(ConfigGeneratorInterface):
    """
    Given Perf Analyzer configuration options, generates Perf Analyzer configs

    All combinations are pregenerated in __init__, but it may return is_done==true
    earlier depending on results that it receives
    """

    def __init__(self, cli_config, model_name, model_perf_analyzer_flags,
                 model_parameters, early_exit_enable):
        """
        Parameters
        ----------
        cli_config: ConfigCommandProfile
            CLI Configuration Options

        model_name: string
            The model name to profile

        model_perf_analyzer_flags: Dict
            custom perf analyzer configuration

        model_parameters: Dict
            model constraints for batch_sizes and/or concurrency

        early_exit_enable: Bool
            If true, this class can early exit during search of concurrency
        """

        self._early_exit_enable = early_exit_enable

        # All configs are pregenerated in _configs[][]
        # Indexed as follows:
        #    _configs[_curr_config_index][_curr_concurrency_index]
        #
        self._curr_concurrency_index = 0
        self._curr_config_index = 0
        self._configs = []
        self._concurrency_warning_printed = False

        # Flag to indicate we have started to return results
        #
        self._generator_started = False

        self._last_results = ["valid"]
        self._all_results = []

        self._model_name = model_name
        self._perf_analyzer_flags = model_perf_analyzer_flags

        self._batch_sizes = sorted(model_parameters['batch_sizes'])
        self._concurrencies = self._create_concurrency_list(
            cli_config, model_parameters)

        self._cli_config = cli_config

        self._generate_perf_configs()

    def _is_done(self):
        """ Returns true if this generator is done generating configs """
        return self._generator_started and self._done_walking()

    def get_configs(self):
        """ Returns the next generated config """
        while True:
            if self._is_done():
                break

            self._generator_started = True
            config = self._configs[self._curr_config_index][
                self._curr_concurrency_index]
            yield (config)

            if self._last_results_erroneous():
                break

            self._step()

    def set_last_results(self, measurements):
        """
        Given the results from the last PerfAnalyzerConfig, make decisions
        about future configurations to generate

        Parameters
        ----------
        measurements: List of Measurements from the last run(s)
        """

        # Remove 'NONE' cases, and find single max measurement from the list
        measurements = [m for m in measurements if m]

        measurement = [max(measurements)] if measurements else [None]

        self._last_results = measurement
        self._all_results.extend(measurement)

    def _create_concurrency_list(self, cli_config, model_parameters):
        if model_parameters['concurrency']:
            return sorted(model_parameters['concurrency'])
        elif cli_config.run_config_search_disable:
            return [1]
        else:
            return utils.generate_doubled_list(
                cli_config.run_config_search_min_concurrency,
                cli_config.run_config_search_max_concurrency)

    def _generate_perf_configs(self):
        perf_config_non_concurrency_params = self._create_non_concurrency_perf_config_params(
        )

        for params in utils.generate_parameter_combinations(
                perf_config_non_concurrency_params):
            configs_with_concurrency = []
            for concurrency in self._concurrencies:
                new_perf_config = PerfAnalyzerConfig()

                new_perf_config.update_config_from_profile_config(
                    self._model_name, self._cli_config)

                new_perf_config.update_config(params)
                new_perf_config.update_config(
                    {'concurrency-range': concurrency})

                # User provided flags can override the search parameters
                new_perf_config.update_config(self._perf_analyzer_flags)

                configs_with_concurrency.append(new_perf_config)
            self._configs.append(configs_with_concurrency)

    def _create_non_concurrency_perf_config_params(self):
        perf_config_params = {
            'batch-size': self._batch_sizes,
        }

        return perf_config_params

    def _step(self):
        self._step_concurrency()

        if self._done_walking_concurrencies():
            self._reset_concurrencies()
            self._step_config()

    def _step_config(self):
        self._curr_config_index += 1

    def _reset_concurrencies(self):
        self._curr_concurrency_index = 0
        self._concurrency_warning_printed = False
        self._all_results = []

    def _step_concurrency(self):
        self._curr_concurrency_index += 1

    def _done_walking(self):
        return len(self._configs) == self._curr_config_index

    def _done_walking_concurrencies(self):
        if len(self._concurrencies) == self._curr_concurrency_index:
            return True
        if self._early_exit_enable and not self._throughput_gain_valid():
            if not self._concurrency_warning_printed:
                logger.info(
                    "No longer increasing concurrency as throughput has plateaued"
                )
                self._concurrency_warning_printed = True
            return True
        return False

    def _last_results_erroneous(self):
        return self._last_results is None or self._last_results[-1] is None

    def _throughput_gain_valid(self):
        """ Check if any of the last X results resulted in valid gain """

        if len(self._all_results) < THROUGHPUT_MINIMUM_CONSECUTIVE_TRIES:
            return True

        valid_gains = [self._calculate_throughput_gain(x) > THROUGHPUT_MINIMUM_GAIN \
                       for x in range(1,THROUGHPUT_MINIMUM_CONSECUTIVE_TRIES)
                      ]
        return True in valid_gains

    def _calculate_throughput_gain(self, reverse_index):
        """
        Given a reverse index, calculate the throughput gain at that index when
        indexing from the back of the results list, when compared to its previous
        results

        For example, setting reverse_index=1 will calculate the gain for the last
        two results in the list (indexes -2 and -1)
        """
        before_index = -(reverse_index + 1)
        after_index = -reverse_index
        throughput_before = self._get_throughput(
            self._all_results[before_index])
        throughput_after = self._get_throughput(self._all_results[after_index])
        gain = (throughput_after - throughput_before) / throughput_before
        return gain

    def _get_throughput(self, measurement):
        return measurement.get_non_gpu_metric_value('perf_throughput')
