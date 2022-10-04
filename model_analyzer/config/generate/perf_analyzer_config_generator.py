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

from typing import List, Generator, Optional

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile

from .config_generator_interface import ConfigGeneratorInterface
from .generator_utils import GeneratorUtils as utils

from model_analyzer.constants import LOGGER_NAME, THROUGHPUT_MINIMUM_GAIN, THROUGHPUT_MINIMUM_CONSECUTIVE_CONCURRENCY_TRIES, THROUGHPUT_MINIMUM_CONSECUTIVE_BATCH_SIZE_TRIES
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

import logging

logger = logging.getLogger(LOGGER_NAME)


class PerfAnalyzerConfigGenerator(ConfigGeneratorInterface):
    """
    Given Perf Analyzer configuration options, generates Perf Analyzer configs

    All combinations are pregenerated in __init__, but it may return is_done==true
    earlier depending on results that it receives
    """

    def __init__(self, cli_config: ConfigCommandProfile, model_name: str,
                 model_perf_analyzer_flags: dict, model_parameters: dict,
                 early_exit_enable: bool) -> None:
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
        #    _configs[_curr_batch_size_index][_curr_concurrency_index]
        #
        self._curr_concurrency_index = 0
        self._curr_batch_size_index = 0
        self._configs: List[List[PerfAnalyzerConfig]] = []
        self._concurrency_warning_printed = False

        # Flag to indicate we have started to return results
        #
        self._generator_started = False

        self._last_results: List[RunConfigMeasurement] = []
        self._concurrency_results: List[Optional[RunConfigMeasurement]] = []
        self._batch_size_results: List[Optional[RunConfigMeasurement]] = []

        self._model_name = model_name
        self._perf_analyzer_flags = model_perf_analyzer_flags

        self._batch_sizes = sorted(model_parameters['batch_sizes'])
        self._concurrencies = self._create_concurrency_list(
            cli_config, model_parameters)

        self._cli_config = cli_config

        self._generate_perf_configs()

    @staticmethod
    def throughput_gain_valid_helper(
            throughputs: List[Optional[RunConfigMeasurement]],
            min_tries: int = THROUGHPUT_MINIMUM_CONSECUTIVE_CONCURRENCY_TRIES,
            min_gain: float = THROUGHPUT_MINIMUM_GAIN) -> bool:
        if len(throughputs) < min_tries:
            return True

        tputs_in_range = [
            PerfAnalyzerConfigGenerator.get_throughput(throughputs[x])
            for x in range(-min_tries, 0)
        ]

        first = tputs_in_range[0]
        best = max(tputs_in_range)

        gain = (best - first) / first

        return gain > min_gain

    @staticmethod
    def get_throughput(measurement: Optional[RunConfigMeasurement]) -> float:
        if measurement:
            return measurement.get_non_gpu_metric_value('perf_throughput')
        else:
            return 0.0

    def _is_done(self) -> bool:
        """ Returns true if this generator is done generating configs """
        return self._generator_started and self._done_walking()

    def get_configs(self) -> Generator[PerfAnalyzerConfig, None, None]:
        """ Returns the next generated config """
        while True:
            if self._is_done():
                break

            self._generator_started = True
            config = self._configs[self._curr_batch_size_index][
                self._curr_concurrency_index]
            yield (config)

            if self._last_results_erroneous():
                break

            self._step()

    def set_last_results(
            self, measurements: List[Optional[RunConfigMeasurement]]) -> None:
        """
        Given the results from the last PerfAnalyzerConfig, make decisions
        about future configurations to generate

        Parameters
        ----------
        measurements: List of Measurements from the last run(s)
        """

        # Remove 'NONE' cases, and find single max measurement from the list
        valid_measurements = [m for m in measurements if m]

        self._last_results = []
        if valid_measurements:
            measurement = [max(valid_measurements)]

            self._last_results = measurement
            self._concurrency_results.extend(measurement)

    def _create_concurrency_list(self, cli_config: ConfigCommandProfile,
                                 model_parameters: dict) -> List[int]:
        if model_parameters['concurrency']:
            return sorted(model_parameters['concurrency'])
        elif cli_config.run_config_search_disable:
            return [1]
        else:
            return utils.generate_doubled_list(
                cli_config.run_config_search_min_concurrency,
                cli_config.run_config_search_max_concurrency)

    def _generate_perf_configs(self) -> None:
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

    def _create_non_concurrency_perf_config_params(self) -> dict:
        perf_config_params = {
            'batch-size': self._batch_sizes,
        }

        return perf_config_params

    def _step(self) -> None:
        self._step_concurrency()

        if self._done_walking_concurrencies():
            self._add_best_throughput_to_batch_sizes()
            self._reset_concurrencies()
            self._step_batch_size()

    def _add_best_throughput_to_batch_sizes(self) -> None:
        if self._concurrency_results:
            # type is List[Optional[RCM]]
            best = max(self._concurrency_results)  #type: ignore
            self._batch_size_results.append(best)

    def _reset_concurrencies(self) -> None:
        self._curr_concurrency_index = 0
        self._concurrency_warning_printed = False
        self._concurrency_results = []

    def _step_concurrency(self) -> None:
        self._curr_concurrency_index += 1

    def _step_batch_size(self) -> None:
        self._curr_batch_size_index += 1

    def _done_walking(self) -> bool:
        return self._done_walking_batch_sizes()

    def _done_walking_concurrencies(self) -> bool:
        if len(self._concurrencies) == self._curr_concurrency_index:
            return True
        if self._early_exit_enable and not self._concurrency_throughput_gain_valid(
        ):
            if not self._concurrency_warning_printed:
                logger.info(
                    "No longer increasing concurrency as throughput has plateaued"
                )
                self._concurrency_warning_printed = True
            return True
        return False

    def _done_walking_batch_sizes(self) -> bool:
        if len(self._batch_sizes) == self._curr_batch_size_index:
            return True

        if self._early_exit_enable and not self._batch_size_throughput_gain_valid(
        ):

            logger.info(
                "No longer increasing client batch size as throughput has plateaued"
            )

            return True
        return False

    def _last_results_erroneous(self) -> bool:
        return not self._last_results or self._last_results[-1] is None

    def _concurrency_throughput_gain_valid(self) -> bool:
        """ Check if any of the last X concurrency results resulted in valid gain """
        return PerfAnalyzerConfigGenerator.throughput_gain_valid_helper(
            throughputs=self._concurrency_results,
            min_tries=THROUGHPUT_MINIMUM_CONSECUTIVE_CONCURRENCY_TRIES,
            min_gain=THROUGHPUT_MINIMUM_GAIN)

    def _batch_size_throughput_gain_valid(self) -> bool:
        """ Check if any of the last X batch_size results resulted in valid gain """
        return PerfAnalyzerConfigGenerator.throughput_gain_valid_helper(
            throughputs=self._batch_size_results,
            min_tries=THROUGHPUT_MINIMUM_CONSECUTIVE_BATCH_SIZE_TRIES,
            min_gain=THROUGHPUT_MINIMUM_GAIN)
