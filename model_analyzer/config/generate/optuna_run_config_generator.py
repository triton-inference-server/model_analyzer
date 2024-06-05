#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from typing import Any, Dict, Generator, List, Optional, TypeAlias, Union

import optuna

from model_analyzer.config.generate.base_model_config_generator import (
    BaseModelConfigGenerator,
)
from model_analyzer.config.generate.brute_run_config_generator import (
    BruteRunConfigGenerator,
)
from model_analyzer.config.generate.model_profile_spec import ModelProfileSpec
from model_analyzer.config.generate.model_variant_name_manager import (
    ModelVariantNameManager,
)
from model_analyzer.config.generate.search_parameter import (
    ParameterCategory,
    SearchParameter,
)
from model_analyzer.config.generate.search_parameters import SearchParameters
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_defaults import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_RUN_CONFIG_MAX_CONCURRENCY,
)
from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.triton.model.model_config_variant import ModelConfigVariant

from .config_generator_interface import ConfigGeneratorInterface

logger = logging.getLogger(LOGGER_NAME)

TrialObjective: TypeAlias = Union[str | int]
TrialObjectives: TypeAlias = Dict[str, TrialObjective]
ParameterCombo: TypeAlias = Dict[str, Any]


class OptunaRunConfigGenerator(ConfigGeneratorInterface):
    """
    Use Optuna algorithm to create RunConfigs
    """

    # This list represents all possible parameters Optuna can currently search for
    optuna_parameter_list = [
        "batch_sizes",
        "instance_group",
        "concurrency",
        "max_queue_delay_microseconds",
    ]

    # TODO: TMA-1927: Figure out the correct value for this
    NO_MEASUREMENT_SCORE = -1

    def __init__(
        self,
        config: ConfigCommandProfile,
        gpu_count: int,
        models: List[ModelProfileSpec],
        model_variant_name_manager: ModelVariantNameManager,
        search_parameters: Dict[str, SearchParameters],
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        config: ConfigCommandProfile
            Profile configuration information
        gpu_count: Number of gpus in the system
        models: List of ModelProfileSpec
            List of models to profile
        model_variant_name_manager: ModelVariantNameManager
        search_parameters: SearchParameters
            The object that handles the users configuration search parameters
        """
        self._config = config
        self._gpu_count = gpu_count
        self._models = models
        # TODO: TMA-1927: Add support for multi-model
        self._search_parameters = search_parameters[models[0].model_name()]

        self._model_variant_name_manager = model_variant_name_manager

        self._triton_env = BruteRunConfigGenerator.determine_triton_server_env(models)

        self._num_models = len(models)
        self._last_measurement: Optional[RunConfigMeasurement] = None
        self._best_config_name = ""
        self._best_config_score: Optional[float] = None
        self._best_trial_number: Optional[int] = None

        self._c_api_mode = config.triton_launch_mode == "c_api"

        self._done = False

        if seed is not None:
            self._sampler = optuna.samplers.TPESampler(seed=seed)
        else:
            self._sampler = optuna.samplers.TPESampler()

        self._study = optuna.create_study(
            study_name=self._models[0].model_name(),
            direction="maximize",
            sampler=self._sampler,
        )

    def _is_done(self) -> bool:
        return self._done

    def set_last_results(
        self, measurements: List[Optional[RunConfigMeasurement]]
    ) -> None:
        # TODO: TMA-1927: Add support for multi-model
        if measurements[0] is not None:
            self._last_measurement = measurements[0]
        else:
            self._last_measurement = None

    def get_configs(self) -> Generator[RunConfig, None, None]:
        """
        Returns
        -------
        RunConfig
            The next RunConfig generated by this class
        """
        logger.info(
            "Measuring default configuration to establish a baseline measurement"
        )
        default_run_config = self._create_default_run_config()
        yield default_run_config

        self._capture_default_measurement(default_run_config)
        self._set_best_measurement(default_run_config)

        if logging.DEBUG:
            self._print_debug_search_space_info()

        min_configs_to_search = self._determine_minimum_number_of_configs_to_search()
        max_configs_to_search = self._determine_maximum_number_of_configs_to_search()
        # TODO: TMA-1885: Need an early exit strategy
        for trial_number in range(1, max_configs_to_search + 1):
            trial = self._study.ask()
            trial_objectives = self._create_trial_objectives(trial)
            logger.debug(f"Trial {trial_number} of {max_configs_to_search}:")
            run_config = self._create_objective_based_run_config(trial_objectives)
            yield run_config

            score = self._calculate_score()
            self._set_best_measurement(run_config, score)

            if logging.DEBUG:
                self._print_debug_score_info(run_config, score)

            if self._should_terminate_early(min_configs_to_search, trial_number):
                logger.debug("Early termination threshold reached")
                break
            self._study.tell(trial, score)

    def _capture_default_measurement(self, default_run_config: RunConfig) -> None:
        if not self._last_measurement:
            raise TritonModelAnalyzerException(
                "Default configuration did not return a measurement. Please check PA/Tritonserver log files."
            )

        self._default_measurement = self._last_measurement

    def _set_best_measurement(
        self, run_config: RunConfig, score: float = 0, trial_number: int = 0
    ) -> None:
        if self._best_config_score is None or score > self._best_config_score:
            self._best_config_name = run_config.model_variants_name()
            self._best_config_score = score
            self._best_trial_number = trial_number

    def _determine_maximum_number_of_configs_to_search(self) -> int:
        max_trials_based_on_percentage_of_search_space = (
            self._determine_trials_based_on_max_percentage_of_search_space()
        )

        max_configs_to_search = self._decide_max_between_percentage_and_trial_count(
            max_trials_based_on_percentage_of_search_space
        )

        return max_configs_to_search

    def _determine_trials_based_on_max_percentage_of_search_space(self) -> int:
        total_num_of_possible_configs = (
            self._search_parameters.number_of_total_possible_configurations()
        )
        max_trials_based_on_percentage_of_search_space = int(
            total_num_of_possible_configs
            * self._config.max_percentage_of_search_space
            / 100
        )

        return max_trials_based_on_percentage_of_search_space

    def _decide_max_between_percentage_and_trial_count(
        self, max_trials_based_on_percentage_of_search_space: int
    ) -> int:
        # By default we will search based on percentage of search space
        # If the user specifies a number of trials we will use that instead
        # If both are specified we will use the smaller number
        max_trials_set_by_user = self._config.get_config()[
            "optuna_max_trials"
        ].is_set_by_user()
        max_percentage_set_by_user = self._config.get_config()[
            "max_percentage_of_search_space"
        ].is_set_by_user()

        if max_trials_set_by_user and max_percentage_set_by_user:
            if (
                self._config.optuna_max_trials
                < max_trials_based_on_percentage_of_search_space
            ):
                logger.debug(
                    f"Maximum number of trials: {self._config.optuna_max_trials} (optuna_max_trials)"
                )
                max_configs_to_search = self._config.optuna_max_trials
            else:
                logger.debug(
                    f"Maximum number of trials: {max_trials_based_on_percentage_of_search_space} "
                    f"({self._config.max_percentage_of_search_space}% of search space)"
                )
                max_configs_to_search = max_trials_based_on_percentage_of_search_space
        elif max_trials_set_by_user:
            logger.debug(
                f"Maximum number of trials: {self._config.optuna_max_trials} (optuna_max_trials)"
            )
            max_configs_to_search = self._config.optuna_max_trials
        else:
            logger.debug(
                f"Maximum number of trials: {max_trials_based_on_percentage_of_search_space} "
                f"({self._config.max_percentage_of_search_space}% of search space)"
            )
            max_configs_to_search = max_trials_based_on_percentage_of_search_space

        if logging.DEBUG:
            logger.info("")
        return max_configs_to_search

    def _determine_minimum_number_of_configs_to_search(self) -> int:
        min_trials_based_on_percentage_of_search_space = (
            self._determine_trials_based_on_min_percentage_of_search_space()
        )

        min_configs_to_search = self._decide_min_between_percentage_and_trial_count(
            min_trials_based_on_percentage_of_search_space
        )

        return min_configs_to_search

    def _determine_trials_based_on_min_percentage_of_search_space(self) -> int:
        total_num_of_possible_configs = (
            self._search_parameters.number_of_total_possible_configurations()
        )
        min_trials_based_on_percentage_of_search_space = int(
            total_num_of_possible_configs
            * self._config.min_percentage_of_search_space
            / 100
        )

        return min_trials_based_on_percentage_of_search_space

    def _decide_min_between_percentage_and_trial_count(
        self, min_trials_based_on_percentage_of_search_space: int
    ) -> int:
        # By default we will search based on percentage of search space
        # If the user specifies a number of trials we will use that instead
        # If both are specified we will use the larger number
        min_trials_set_by_user = self._config.get_config()[
            "optuna_min_trials"
        ].is_set_by_user()
        min_percentage_set_by_user = self._config.get_config()[
            "min_percentage_of_search_space"
        ].is_set_by_user()

        if min_trials_set_by_user and min_percentage_set_by_user:
            if (
                self._config.optuna_min_trials
                > min_trials_based_on_percentage_of_search_space
            ):
                logger.debug(
                    f"Minimum number of trials: {self._config.optuna_min_trials} (optuna_min_trials)"
                )
                min_configs_to_search = self._config.optuna_min_trials
            else:
                logger.debug(
                    f"Minimum number of trials: {min_trials_based_on_percentage_of_search_space} "
                    f"({self._config.min_percentage_of_search_space}% of search space)"
                )
                min_configs_to_search = min_trials_based_on_percentage_of_search_space
        elif min_trials_set_by_user:
            logger.debug(
                f"Minimum number of trials: {self._config.optuna_min_trials} (optuna_min_trials)"
            )
            min_configs_to_search = self._config.optuna_min_trials
        else:
            logger.debug(
                f"Minimum number of trials: {min_trials_based_on_percentage_of_search_space} "
                f"({self._config.min_percentage_of_search_space}% of search space)"
            )
            min_configs_to_search = min_trials_based_on_percentage_of_search_space

        return min_configs_to_search

    def _create_trial_objectives(self, trial: optuna.Trial) -> TrialObjectives:
        trial_objectives: TrialObjectives = {}
        for parameter_name in OptunaRunConfigGenerator.optuna_parameter_list:
            parameter = self._search_parameters.get_parameter(parameter_name)

            if parameter:
                trial_objectives[parameter_name] = self._create_trial_objective(
                    trial, parameter_name, parameter
                )

        if self._config.use_concurrency_formula:
            trial_objectives["concurrency"] = self._get_objective_concurrency(
                trial_objectives
            )

        return trial_objectives

    def _create_trial_objective(
        self, trial: optuna.Trial, name: str, parameter: SearchParameter
    ) -> TrialObjective:
        if parameter.category is ParameterCategory.INTEGER:
            objective = trial.suggest_int(
                name, parameter.min_range, parameter.max_range
            )
        elif parameter.category is ParameterCategory.EXPONENTIAL:
            objective = int(
                2 ** trial.suggest_int(name, parameter.min_range, parameter.max_range)
            )
        elif parameter.category is ParameterCategory.INT_LIST:
            objective = int(trial.suggest_categorical(name, parameter.enumerated_list))
        elif parameter.category is ParameterCategory.STR_LIST:
            objective = trial.suggest_categorical(name, parameter.enumerated_list)

        return objective

    def _get_objective_concurrency(self, trial_objectives: TrialObjectives) -> int:
        concurrency_formula = (
            2
            * int(trial_objectives["instance_group"])
            * int(trial_objectives["batch_sizes"])
        )
        concurrency = (
            self._config.run_config_search_max_concurrency
            if concurrency_formula > self._config.run_config_search_max_concurrency
            else concurrency_formula
        )

        return concurrency

    def _create_objective_based_run_config(
        self, trial_objectives: TrialObjectives
    ) -> RunConfig:
        param_combo = self._create_parameter_combo(trial_objectives)

        # TODO: TMA-1927: Add support for multi-model
        run_config = RunConfig(self._triton_env)

        model_config_variant = BaseModelConfigGenerator.make_model_config_variant(
            param_combo=param_combo,
            model=self._models[0],
            model_variant_name_manager=self._model_variant_name_manager,
            c_api_mode=self._c_api_mode,
        )

        # TODO: TMA-1927: Add support for multi-model
        model_run_config = self._create_model_run_config(
            model=self._models[0],
            model_config_variant=model_config_variant,
            trial_objectives=trial_objectives,
        )

        run_config.add_model_run_config(model_run_config=model_run_config)

        return run_config

    def _create_parameter_combo(
        self, trial_objectives: TrialObjectives
    ) -> ParameterCombo:
        param_combo: ParameterCombo = {}

        if self._models[0].supports_dynamic_batching():
            param_combo["dynamic_batching"] = []

        # TODO: TMA-1927: Add support for multi-model
        if "instance_group" in trial_objectives:
            kind = "KIND_CPU" if self._models[0].cpu_only() else "KIND_GPU"
            param_combo["instance_group"] = [
                {
                    "count": trial_objectives["instance_group"],
                    "kind": kind,
                }
            ]

        if "batch_sizes" in trial_objectives:
            param_combo["max_batch_size"] = trial_objectives["batch_sizes"]

        if "max_queue_delay_microseconds" in trial_objectives:
            param_combo["dynamic_batching"] = {
                "max_queue_delay_microseconds": trial_objectives[
                    "max_queue_delay_microseconds"
                ]
            }

        return param_combo

    def _calculate_score(self) -> float:
        if self._last_measurement:
            score = self._default_measurement.compare_measurements(  # type: ignore
                self._last_measurement
            )
        else:
            score = OptunaRunConfigGenerator.NO_MEASUREMENT_SCORE

        return score

    def _create_default_run_config(self) -> RunConfig:
        default_run_config = RunConfig(self._triton_env)
        # TODO: TMA-1927: Add support for multi-model
        default_model_run_config = self._create_default_model_run_config(
            self._models[0]
        )
        default_run_config.add_model_run_config(default_model_run_config)

        return default_run_config

    def _create_default_model_run_config(
        self, model: ModelProfileSpec
    ) -> ModelRunConfig:
        default_model_config_variant = (
            BaseModelConfigGenerator.make_model_config_variant(
                param_combo={},
                model=model,
                model_variant_name_manager=self._model_variant_name_manager,
                c_api_mode=self._c_api_mode,
            )
        )

        default_perf_analyzer_config = self._create_default_perf_analyzer_config(
            model, default_model_config_variant.model_config
        )

        default_model_run_config = ModelRunConfig(
            model.model_name(),
            default_model_config_variant,
            default_perf_analyzer_config,
        )

        return default_model_run_config

    def _create_default_perf_analyzer_config(
        self, model: ModelProfileSpec, model_config: ModelConfig
    ) -> PerfAnalyzerConfig:
        default_perf_analyzer_config = PerfAnalyzerConfig()
        default_perf_analyzer_config.update_config_from_profile_config(
            model_config.get_field("name"), self._config
        )

        default_concurrency = self._calculate_default_concurrency(model_config)

        perf_config_params = {
            "batch-size": DEFAULT_BATCH_SIZES,
            "concurrency-range": default_concurrency,
        }
        default_perf_analyzer_config.update_config(perf_config_params)

        default_perf_analyzer_config.update_config(model.perf_analyzer_flags())

        return default_perf_analyzer_config

    def _calculate_default_concurrency(self, model_config: ModelConfig) -> int:
        default_max_batch_size = model_config.max_batch_size()
        default_instance_count = model_config.instance_group_count(
            system_gpu_count=self._gpu_count
        )
        default_concurrency = 2 * default_max_batch_size * default_instance_count

        return default_concurrency

    def _create_model_run_config(
        self,
        model: ModelProfileSpec,
        model_config_variant: ModelConfigVariant,
        trial_objectives: TrialObjectives,
    ) -> ModelRunConfig:
        perf_analyzer_config = self._create_perf_analyzer_config(
            model.model_name(), model, int(trial_objectives["concurrency"])
        )
        model_run_config = ModelRunConfig(
            model.model_name(), model_config_variant, perf_analyzer_config
        )

        return model_run_config

    def _create_perf_analyzer_config(
        self,
        model_name: str,
        model: ModelProfileSpec,
        concurrency: int,
    ) -> PerfAnalyzerConfig:
        perf_analyzer_config = PerfAnalyzerConfig()

        perf_analyzer_config.update_config_from_profile_config(model_name, self._config)

        # TODO: TMA-1934 add support for user specifying a range of client batch sizes
        perf_config_params = {
            "batch-size": DEFAULT_BATCH_SIZES,
            "concurrency-range": concurrency,
        }
        perf_analyzer_config.update_config(perf_config_params)

        perf_analyzer_config.update_config(model.perf_analyzer_flags())
        return perf_analyzer_config

    def _should_terminate_early(
        self, min_configs_to_search: int, trial_number: int
    ) -> bool:
        number_of_trials_since_best = trial_number - self._best_trial_number  # type: ignore
        if trial_number < min_configs_to_search:
            should_terminate_early = False
        elif number_of_trials_since_best >= self._config.optuna_early_exit_threshold:
            should_terminate_early = True
        else:
            should_terminate_early = False

        return should_terminate_early

    def _print_debug_search_space_info(self) -> None:
        logger.info("")
        logger.debug(
            f"Number of configs in search space: {self._search_parameters.number_of_total_possible_configurations()}"
        )

        for name in self._search_parameters.get_search_parameters():
            logger.debug(self._search_parameters.print_info(name))

        logger.info("")

    def _print_debug_score_info(
        self,
        run_config: RunConfig,
        score: float,
    ) -> None:
        if score != OptunaRunConfigGenerator.NO_MEASUREMENT_SCORE:
            logger.debug(
                f"Objective score for {run_config.model_variants_name()}: {int(score * 100)} --- "  # type: ignore
                f"Best: {self._best_config_name} ({int(self._best_config_score * 100)})"  # type: ignore
            )

        logger.info("")
