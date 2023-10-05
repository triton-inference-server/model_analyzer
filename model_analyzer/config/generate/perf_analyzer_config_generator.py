#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging
import tempfile
from copy import deepcopy
from typing import Dict, Generator, List, Optional, Tuple

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_defaults import DEFAULT_INPUT_JSON_PATH
from model_analyzer.constants import (
    LOGGER_NAME,
    THROUGHPUT_MINIMUM_CONSECUTIVE_NON_PARAMETER_TRIES,
    THROUGHPUT_MINIMUM_CONSECUTIVE_PARAMETER_TRIES,
    THROUGHPUT_MINIMUM_GAIN,
)
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

from .config_generator_interface import ConfigGeneratorInterface
from .generator_utils import GeneratorUtils as utils

logger = logging.getLogger(LOGGER_NAME)


class PerfAnalyzerConfigGenerator(ConfigGeneratorInterface):
    """
    Given Perf Analyzer configuration options, generates Perf Analyzer configs

    All combinations are pregenerated in __init__, but it may return is_done==true
    earlier depending on results that it receives
    """

    def __init__(
        self,
        cli_config: ConfigCommandProfile,
        model_name: str,
        model_perf_analyzer_flags: dict,
        model_parameters: dict,
        early_exit_enable: bool,
    ) -> None:
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
            model constraints for batch sizes, concurrency, request rate, prompt length, etc..

        early_exit_enable: Bool
            If true, this class can early exit during search of concurrency/request rate
        """

        self._early_exit_enable = early_exit_enable

        # All configs are pregenerated in _configs[][]
        # Indexed as follows:
        #    _configs[_curr_non_parameter_index][_curr_parameter_index]
        #
        self._curr_non_parameter_index = 0
        self._curr_parameter_index = 0
        self._configs: List[List[PerfAnalyzerConfig]] = []
        self._parameter_warning_printed = False

        # Flag to indicate we have started to return results
        #
        self._generator_started = False

        self._last_results: List[RunConfigMeasurement] = []
        self._parameter_results: List[Optional[RunConfigMeasurement]] = []
        self._non_parameter_results: List[Optional[RunConfigMeasurement]] = []

        self._model_name = model_name
        self._cli_config = cli_config

        self._perf_analyzer_flags = self._set_perf_analyzer_flags(
            model_perf_analyzer_flags
        )

        self._llm_input_dict = self._create_input_dict(model_perf_analyzer_flags)
        self._model_parameters = model_parameters
        self._parameters = self._create_parameter_list()

        self._batch_sizes = sorted(model_parameters["batch_sizes"])
        self._prompt_lengths = self._create_prompt_length_list()
        self._max_token_counts = self._create_max_token_count_list()

        self._perf_config_non_parameter_values = (
            self._create_non_parameter_perf_config_values()
        )
        self._non_parameter_count = len(
            utils.generate_parameter_combinations(
                self._perf_config_non_parameter_values
            )
        )

        self._input_json_filename = DEFAULT_INPUT_JSON_PATH + "/input-data.json"

        self._generate_perf_configs()

    @staticmethod
    def throughput_gain_valid_helper(
        throughputs: List[Optional[RunConfigMeasurement]],
        min_tries: int = THROUGHPUT_MINIMUM_CONSECUTIVE_PARAMETER_TRIES,
        min_gain: float = THROUGHPUT_MINIMUM_GAIN,
    ) -> bool:
        if len(throughputs) < min_tries:
            return True

        throughputs_in_range = [
            PerfAnalyzerConfigGenerator.get_throughput(throughputs[x])
            for x in range(-min_tries, 0)
        ]

        first = throughputs_in_range[0]
        best = max(throughputs_in_range)

        gain = (best - first) / first

        return gain > min_gain

    @staticmethod
    def get_throughput(measurement: Optional[RunConfigMeasurement]) -> float:
        if measurement:
            return measurement.get_non_gpu_metric_value("perf_throughput")
        else:
            return 0.0

    def _is_done(self) -> bool:
        """Returns true if this generator is done generating configs"""
        return self._generator_started and self._done_walking()

    def get_configs(self) -> Generator[PerfAnalyzerConfig, None, None]:
        """Returns the next generated config"""
        while True:
            if self._is_done():
                break

            self._generator_started = True
            config = self._configs[self._curr_non_parameter_index][
                self._curr_parameter_index
            ]
            yield (config)

            if self._last_results_erroneous():
                break

            self._step()

    def set_last_results(
        self, measurements: List[Optional[RunConfigMeasurement]]
    ) -> None:
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
            self._parameter_results.extend(measurement)

    def _set_perf_analyzer_flags(self, model_perf_analyzer_flags: dict) -> dict:
        # For LLM models we will be creating custom input data based on prompt length
        if self._cli_config.is_llm_model():
            perf_analyzer_flags = deepcopy(model_perf_analyzer_flags)
            perf_analyzer_flags.pop("input-data")
            return perf_analyzer_flags
        else:
            return model_perf_analyzer_flags

    def _create_input_dict(self, model_perf_analyzer_flags: dict) -> dict:
        if self._cli_config.is_llm_model():
            with open(model_perf_analyzer_flags["input-data"], "r") as f:
                input_dict = json.load(f)

            return input_dict
        else:
            return {}

    def _modify_prompt_in_input_dict(self, prompt_length: int) -> Dict:
        modified_input_dict = deepcopy(self._llm_input_dict)

        modified_prompt = ["hi"] * prompt_length
        modified_input_dict["data"][0]["PROMPT"] = modified_prompt

        return modified_input_dict

    def _create_parameter_list(self) -> List[int]:
        # The two possible parameters are request rate or concurrency
        # Concurrency is the default and will be used unless the user specifies
        # request rate, either as a model parameter or a config option
        if self._cli_config.is_request_rate_specified(self._model_parameters):
            return self._create_request_rate_list()
        else:
            return self._create_concurrency_list()

    def _create_request_rate_list(self) -> List[int]:
        if self._model_parameters["request_rate"]:
            return sorted(self._model_parameters["request_rate"])
        elif self._cli_config.run_config_search_disable:
            return [1]
        else:
            return utils.generate_doubled_list(
                self._cli_config.run_config_search_min_request_rate,
                self._cli_config.run_config_search_max_request_rate,
            )

    def _create_concurrency_list(self) -> List[int]:
        if self._model_parameters["concurrency"]:
            return sorted(self._model_parameters["concurrency"])
        elif self._cli_config.run_config_search_disable:
            return [1]
        else:
            return utils.generate_doubled_list(
                self._cli_config.run_config_search_min_concurrency,
                self._cli_config.run_config_search_max_concurrency,
            )

    def _create_prompt_length_list(self) -> List[int]:
        if not self._cli_config.is_llm_model():
            return []

        if self._model_parameters["prompt_length"]:
            return sorted(self._model_parameters["prompt_length"])
        elif self._cli_config.run_config_search_disable:
            return [1]
        else:
            return utils.generate_doubled_list(
                self._cli_config.run_config_search_min_prompt_length,
                self._cli_config.run_config_search_max_prompt_length,
            )

    def _create_max_token_count_list(self) -> List[int]:
        if not self._cli_config.is_llm_model():
            return []

        if self._model_parameters["max_token_count"]:
            return sorted(self._model_parameters["max_token_count"])
        elif self._cli_config.run_config_search_disable:
            return [1]
        else:
            return utils.generate_doubled_list(
                self._cli_config.run_config_search_min_token_count,
                self._cli_config.run_config_search_max_token_count,
            )

    def _generate_perf_configs(self) -> None:
        for (
            unmodified_non_parameter_combination
        ) in utils.generate_parameter_combinations(
            self._perf_config_non_parameter_values
        ):
            all_perf_configs_for_a_given_parameter = []
            for parameter in self._parameters:
                perf_config = self._create_base_perf_config()

                (
                    prompt_length,
                    modified_non_parameter_combination,
                ) = self._extract_prompt_length(unmodified_non_parameter_combination)

                self._update_perf_config_based_on_non_parameter_combination(
                    perf_config, modified_non_parameter_combination
                )
                self._update_perf_config_based_on_parameter(perf_config, parameter)
                self._update_perf_config_based_on_perf_analyzer_flags(perf_config)
                self._update_perf_config_for_llm_model(perf_config, prompt_length)

                all_perf_configs_for_a_given_parameter.append(perf_config)
            self._configs.append(all_perf_configs_for_a_given_parameter)

    def _create_base_perf_config(self) -> PerfAnalyzerConfig:
        perf_config = PerfAnalyzerConfig()
        perf_config.update_config_from_profile_config(
            self._model_name, self._cli_config
        )

        return perf_config

    def _extract_prompt_length(
        self, unmodified_parameter_combination: List[Dict]
    ) -> Tuple[int, List[Dict]]:
        if self._cli_config.is_llm_model():
            modified_parameter_combination = deepcopy(unmodified_parameter_combination)
            prompt_length = modified_parameter_combination.pop("prompt-length")

            return prompt_length, modified_parameter_combination
        else:
            return None, unmodified_parameter_combination

    def _update_perf_config_based_on_non_parameter_combination(
        self, perf_config: PerfAnalyzerConfig, non_parameter_combination: Dict
    ) -> None:
        perf_config.update_config(non_parameter_combination)

    def _update_perf_config_based_on_perf_analyzer_flags(
        self, perf_config: PerfAnalyzerConfig
    ) -> None:
        perf_config.update_config(self._perf_analyzer_flags)

    def _update_perf_config_based_on_parameter(
        self, perf_config: PerfAnalyzerConfig, parameter: int
    ) -> None:
        if self._cli_config.is_llm_model():
            perf_config.update_config({"periodic-concurrency-range": parameter})
        elif self._cli_config.is_request_rate_specified(self._model_parameters):
            perf_config.update_config({"request-rate-range": parameter})
        else:
            perf_config.update_config({"concurrency-range": parameter})

    def _update_perf_config_for_llm_model(
        self, perf_config: PerfAnalyzerConfig, prompt_length: int
    ) -> None:
        if not self._cli_config.is_llm_model():
            return

        modified_input_dict = self._modify_prompt_in_input_dict(prompt_length)
        self._write_modified_input_dict_to_file(modified_input_dict)

        perf_config.update_config({"input-data": self._input_json_filename})

    def _write_modified_input_dict_to_file(self, modified_input_dict: Dict) -> None:
        temp_input_data = open(self._input_json_filename, "w")
        json.dump(modified_input_dict, temp_input_data)
        temp_input_data.close()

    def _create_non_parameter_perf_config_values(self) -> dict:
        perf_config_values = {
            "batch-size": self._batch_sizes,
        }

        if self._cli_config.is_llm_model():
            perf_config_values["max-token-count"] = self._max_token_counts
            perf_config_values["prompt-length"] = self._prompt_lengths

        return perf_config_values

    def _step(self) -> None:
        self._step_parameter()

        if self._done_walking_parameters():
            self._add_best_throughput_to_non_parameter_results()
            self._reset_parameters()
            self._step_non_parameter()

    def _add_best_throughput_to_non_parameter_results(self) -> None:
        if self._parameter_results:
            # type is List[Optional[RCM]]
            best = max(self._parameter_results)  # type: ignore
            self._non_parameter_results.append(best)

    def _reset_parameters(self) -> None:
        self._curr_parameter_index = 0
        self._parameter_warning_printed = False
        self._parameter_results = []

    def _step_parameter(self) -> None:
        self._curr_parameter_index += 1

    def _step_non_parameter(self) -> None:
        self._curr_non_parameter_index += 1

    def _done_walking(self) -> bool:
        return self._done_walking_non_parameters()

    def _done_walking_parameters(self) -> bool:
        if len(self._parameters) == self._curr_parameter_index:
            return True
        if self._early_exit_enable and not self._parameter_throughput_gain_valid():
            if not self._parameter_warning_printed:
                if self._cli_config.is_request_rate_specified(self._model_parameters):
                    logger.info(
                        "No longer increasing request rate as throughput has plateaued"
                    )
                else:
                    logger.info(
                        "No longer increasing concurrency as throughput has plateaued"
                    )
                self._parameter_warning_printed = True
            return True
        return False

    def _done_walking_non_parameters(self) -> bool:
        if self._non_parameter_count == self._curr_non_parameter_index:
            return True

        if self._early_exit_enable and not self._non_parameter_throughput_gain_valid():
            logger.info(
                "No longer increasing client batch size as throughput has plateaued"
            )

            return True
        return False

    def _last_results_erroneous(self) -> bool:
        return not self._last_results or self._last_results[-1] is None

    def _parameter_throughput_gain_valid(self) -> bool:
        """Check if any of the last X parameter results resulted in valid gain"""
        return PerfAnalyzerConfigGenerator.throughput_gain_valid_helper(
            throughputs=self._parameter_results,
            min_tries=THROUGHPUT_MINIMUM_CONSECUTIVE_PARAMETER_TRIES,
            min_gain=THROUGHPUT_MINIMUM_GAIN,
        )

    def _non_parameter_throughput_gain_valid(self) -> bool:
        """Check if any of the last X non-parameter results resulted in valid gain"""
        return PerfAnalyzerConfigGenerator.throughput_gain_valid_helper(
            throughputs=self._non_parameter_results,
            min_tries=THROUGHPUT_MINIMUM_CONSECUTIVE_NON_PARAMETER_TRIES,
            min_gain=THROUGHPUT_MINIMUM_GAIN,
        )
