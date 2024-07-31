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
from typing import Generator, List, Optional

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.result.parameter_search import ParameterSearch
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

logger = logging.getLogger(LOGGER_NAME)


class ConcurrencySweeper:
    """
    Sweeps concurrency for the top-N model configs
    """

    def __init__(
        self,
        config: ConfigCommandProfile,
        result_manager: ResultManager,
    ):
        self._config = config
        self._result_manager = result_manager
        self._last_measurement: Optional[RunConfigMeasurement] = None

    def set_last_results(
        self, measurements: List[Optional[RunConfigMeasurement]]
    ) -> None:
        self._last_measurement = measurements[-1]

    def get_configs(self) -> Generator[RunConfig, None, None]:
        """
        A generator which creates RunConfigs based on sweeping
        concurrency over the top-N models
        """
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
                    run_config = self._create_run_config(run_config, concurrency)
                    yield run_config
                    parameter_search.add_run_config_measurement(self._last_measurement)

    def _create_run_config(self, run_config: RunConfig, concurrency: int) -> RunConfig:
        for model_run_config in run_config.model_run_configs():
            perf_config = model_run_config.perf_config()
            perf_config.update_config({"concurrency-range": concurrency})

        return run_config
