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
from copy import deepcopy
from typing import Dict, Generator, List, Optional

from model_analyzer.config.generate.model_profile_spec import ModelProfileSpec
from model_analyzer.config.generate.model_variant_name_manager import (
    ModelVariantNameManager,
)
from model_analyzer.config.generate.optuna_run_config_generator import (
    OptunaRunConfigGenerator,
)
from model_analyzer.config.generate.search_parameters import SearchParameters
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.result.parameter_search import ParameterSearch
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

from .config_generator_interface import ConfigGeneratorInterface

logger = logging.getLogger(LOGGER_NAME)


class OptunaPlusConcurrencySweepRunConfigGenerator(ConfigGeneratorInterface):
    """
    First run OptunaConfigGenerator for an Optuna search, then use
    ParameterSearch for a concurrency sweep + binary search of the default
    and Top N results
    """

    def __init__(
        self,
        config: ConfigCommandProfile,
        gpu_count: int,
        models: List[ModelProfileSpec],
        result_manager: ResultManager,
        model_variant_name_manager: ModelVariantNameManager,
        search_parameters: Dict[str, SearchParameters],
    ):
        """
        Parameters
        ----------
        config: ConfigCommandProfile
            Profile configuration information
        gpu_count: Number of gpus in the system
        models: List of ModelProfileSpec
            List of models to profile
        result_manager: ResultManager
            The object that handles storing and sorting the results from the perf analyzer
        model_variant_name_manager: ModelVariantNameManager
            Maps model variants to config names
        search_parameters: SearchParameters
            The object that handles the users configuration search parameters
        """
        self._config = config
        self._gpu_count = gpu_count
        self._models = models
        self._result_manager = result_manager
        self._model_variant_name_manager = model_variant_name_manager
        self._search_parameters = search_parameters

    def set_last_results(
        self, measurements: List[Optional[RunConfigMeasurement]]
    ) -> None:
        self._last_measurement = measurements[-1]
        self._rcg.set_last_results(measurements)

    def get_configs(self) -> Generator[RunConfig, None, None]:
        """
        Returns
        -------
        RunConfig
            The next RunConfig generated by this class
        """

        logger.info("")
        logger.info("Starting Optuna mode search to find optimal configs")
        logger.info("")
        yield from self._execute_optuna_search()
        logger.info("")
        if self._config.concurrency_sweep_disable:
            logger.info("Done with Optuna mode search.")
        else:
            logger.info(
                "Done with Optuna mode search. Gathering concurrency sweep measurements for reports"
            )
            logger.info("")
            yield from self._sweep_concurrency_over_top_results()
            logger.info("")
            logger.info("Done gathering concurrency sweep measurements for reports")
        logger.info("")

    def _execute_optuna_search(self) -> Generator[RunConfig, None, None]:
        self._rcg: ConfigGeneratorInterface = self._create_optuna_run_config_generator()

        yield from self._rcg.get_configs()

    def _create_optuna_run_config_generator(self) -> OptunaRunConfigGenerator:
        return OptunaRunConfigGenerator(
            config=self._config,
            gpu_count=self._gpu_count,
            models=self._models,
            model_variant_name_manager=self._model_variant_name_manager,
            search_parameters=self._search_parameters,
        )

    def _sweep_concurrency_over_top_results(self) -> Generator[RunConfig, None, None]:
        for model_name in self._result_manager.get_model_names():
            top_results = self._result_manager.top_n_results(
                model_name=model_name,
                n=self._config.num_configs_per_model,
                include_default=True,
            )

            for result in top_results:
                run_config = deepcopy(result.run_config())
                parameter_search = ParameterSearch(self._config)
                for concurrency in parameter_search.search_parameters():
                    run_config = self._set_concurrency(run_config, concurrency)
                    yield run_config
                    parameter_search.add_run_config_measurement(self._last_measurement)

    def _set_concurrency(self, run_config: RunConfig, concurrency: int) -> RunConfig:
        for model_run_config in run_config.model_run_configs():
            perf_config = model_run_config.perf_config()
            perf_config.update_config({"concurrency-range": concurrency})

        return run_config
