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

from typing import List, Optional, Generator

from .config_generator_interface import ConfigGeneratorInterface

from model_analyzer.config.generate.search_config import SearchConfig
from model_analyzer.config.generate.brute_run_config_generator import BruteRunConfigGenerator
from model_analyzer.config.generate.quick_run_config_generator import QuickRunConfigGenerator
from model_analyzer.config.generate.model_variant_name_manager import ModelVariantNameManager
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.generate.model_profile_spec import ModelProfileSpec
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from model_analyzer.result.run_config_result import RunConfigResult

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.config.input.config_defaults import DEFAULT_RUN_CONFIG_MIN_CONCURRENCY, DEFAULT_RUN_CONFIG_MAX_CONCURRENCY

from copy import deepcopy

import logging

logger = logging.getLogger(LOGGER_NAME)


class QuickPlusConcurrencySweepRunConfigGenerator(ConfigGeneratorInterface):
    """
    First run QuickRunConfigGenerator for a hill climbing search, then use 
    Brute for a concurrency sweep of the default and Top N results
    """

    def __init__(self, search_config: SearchConfig,
                 config: ConfigCommandProfile, gpus: List[GPUDevice],
                 models: List[ModelProfileSpec], client: TritonClient,
                 result_manager: ResultManager,
                 model_variant_name_manager: ModelVariantNameManager):
        """
        Parameters
        ----------
        search_config: SearchConfig
            Defines parameters and dimensions for the search
        config: ConfigCommandProfile
            Profile configuration information
        gpus: List of GPUDevices
        models: List of ConfigModelProfileSpec
            List of models to profile
        client: TritonClient
        result_manager: ResultManager
            The object that handles storing and sorting the results from the perf analyzer
        model_variant_name_manager: ModelVariantNameManager
            Maps model variants to config names
        
        model_variant_name_manager: ModelVariantNameManager
        """
        self._search_config = search_config
        self._config = config
        self._gpus = gpus
        self._models = models
        self._client = client
        self._result_manager = result_manager
        self._model_variant_name_manager = model_variant_name_manager

    def set_last_results(self,
                         measurements: List[Optional[RunConfigMeasurement]]):
        self._rcg.set_last_results(measurements)

    def get_configs(self) -> Generator[RunConfig, None, None]:
        """
        Returns
        -------
        RunConfig
            The next RunConfig generated by this class
        """

        logger.info("")
        logger.info("Starting quick mode search to find optimal configs")
        logger.info("")
        yield from self._execute_quick_search()
        logger.info("")
        logger.info(
            "Done with quick mode search. Gathering concurrency sweep measurements for reports"
        )
        logger.info("")
        yield from self._sweep_concurrency_over_top_results()
        logger.info("")
        logger.info("Done gathering concurrency sweep measurements for reports")
        logger.info("")

    def _execute_quick_search(self):
        self._rcg = self._create_quick_run_config_generator()

        yield from self._rcg.get_configs()

    def _create_quick_run_config_generator(self) -> QuickRunConfigGenerator:
        return QuickRunConfigGenerator(
            search_config=self._search_config,
            config=self._config,
            gpus=self._gpus,
            models=self._models,
            client=self._client,
            model_variant_name_manager=self._model_variant_name_manager)

    def _sweep_concurrency_over_top_results(self):
        top_results = self._result_manager.top_n_results(
            n=self._config.num_configs_per_model)

        for count, result in enumerate(top_results):
            new_config = self._create_new_config_command_profile(result)
            self._rcg = self._create_brute_run_config_generator(
                new_config, skip_default_config=(count != 0))

            yield from self._rcg.get_configs()

    def _create_new_config_command_profile(
            self, result: RunConfigResult) -> ConfigCommandProfile:
        new_config = deepcopy(self._config)

        new_config = self._set_search_mode(new_config)
        new_config = self._set_parameters(result, new_config)

        return new_config

    def _create_brute_run_config_generator(
            self, new_config: ConfigCommandProfile,
            skip_default_config: bool) -> BruteRunConfigGenerator:
        return BruteRunConfigGenerator(
            config=new_config,
            gpus=self._gpus,
            models=self._models,
            client=self._client,
            model_variant_name_manager=self._model_variant_name_manager,
            skip_default_config=skip_default_config)

    def _set_search_mode(self,
                         config: ConfigCommandProfile) -> ConfigCommandProfile:
        config.run_config_search_mode = 'brute'
        config.run_config_search_disable = False
        config.early_exit_enable = True

        return config

    def _set_parameters(self, result: RunConfigResult,
                        config: ConfigCommandProfile) -> ConfigCommandProfile:
        batch_size = self._find_batch_size(result)
        config = self._set_batch_size(config, batch_size)

        instance_count = self._find_instance_count(result)
        config = self._set_instance_count(config, instance_count)

        config = self._set_concurrency(config)

        return config

    # TODO: TMA-879 - This approach isn't going to work for multi-model
    # We will need to create a yaml config to set each models
    # batch size/instances seperately
    def _find_batch_size(self, result: RunConfigResult) -> int:
        mc = result.run_config().model_run_configs()[0].model_config(
        ).get_config()

        batch_size = 1
        if 'max_batch_size' in mc:
            batch_size = mc['max_batch_size']
        return batch_size

    def _find_instance_count(self, result: RunConfigResult) -> int:
        return result.run_config().model_run_configs()[0].model_config(
        ).get_config()['instance_group'][0]['count']

    def _set_batch_size(self, config: ConfigCommandProfile,
                        batch_size: int) -> ConfigCommandProfile:
        config.run_config_search_min_model_batch_size = batch_size
        config.run_config_search_max_model_batch_size = batch_size

        return config

    def _set_instance_count(self, config: ConfigCommandProfile,
                            instance_count: int) -> ConfigCommandProfile:
        config.run_config_search_min_instance_count = instance_count
        config.run_config_search_max_instance_count = instance_count

        return config

    def _set_concurrency(self,
                         config: ConfigCommandProfile) -> ConfigCommandProfile:
        config.run_config_search_min_concurrency = DEFAULT_RUN_CONFIG_MIN_CONCURRENCY
        config.run_config_search_max_concurrency = DEFAULT_RUN_CONFIG_MAX_CONCURRENCY

        return config
