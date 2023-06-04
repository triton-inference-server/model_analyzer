# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, Optional

from model_analyzer.constants import LOGGER_NAME, INVALID_MEASUREMENT_THRESHOLD
from model_analyzer.config.generate.run_config_generator_factory import RunConfigGeneratorFactory
from .model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.config.generate.model_variant_name_manager import ModelVariantNameManager

from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.triton.server.server import TritonServer
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.config.input.objects.config_model_profile_spec import ConfigModelProfileSpec
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

import logging

logger = logging.getLogger(LOGGER_NAME)


class ModelManager:
    """
    This class handles the search for, creation of, and execution of run configs.
    It also records the best results for each model.
    """

    def __init__(self, config: ConfigCommandProfile, gpus: List[GPUDevice],
                 client: TritonClient, server: TritonServer,
                 metrics_manager: MetricsManager, result_manager: ResultManager,
                 state_manager: AnalyzerStateManager,
                 constraint_manager: ConstraintManager):
        """
        Parameters
        ----------
        config:ConfigCommandProfile
            The config for the model analyzer
        gpus: List of GPUDevice
        client: TritonClient
            The client handle used to send requests to Triton
        server: TritonServer
            The server handle used to start and stop Triton instances
        metrics_manager: MetricsManager
            The object that handles launching perf analyzer instances and profiling.
        result_manager: ResultManager
            The object that handles storing and sorting the results from the perf analyzer
        state_manager: AnalyzerStateManager
            The object that handles serializing the state of the analyzer and saving.
        constraint_manager: ConstraintManager
            The object that handles processing and applying
            constraints on a given measurements
        """

        self._config = config
        self._gpus = gpus
        self._client = client
        self._server = server
        self._metrics_manager = metrics_manager
        self._result_manager = result_manager
        self._state_manager = state_manager
        self._constraint_manager = constraint_manager

        if state_manager.starting_fresh_run():
            self._init_state()

        self._failed_measurement_attempts = 0
        self._received_measurement_values_from_pa = False

        self._model_variant_name_manager = ModelVariantNameManager.from_dict(
            self._state_manager.get_state_variable(
                'ModelManager.model_variant_name_manager'))

    def run_models(self, models: List[ConfigModelProfileSpec]) -> None:
        """
        Generates configs, runs inferences, gets
        measurements for a list of models

        Parameters
        ----------
        models : List of ConfigModelProfileSpec
            The models to run
        """

        # Note: this is not done in config_command, because there isn't a ModelConfig yet,
        # so we cannot determine if the model is an ensemble
        self._check_for_ensemble_model_incompatibility(models)

        self._metrics_manager.start_new_model()

        # Save the global server config and update the server's config for this model run
        server_config_copy = self._server.config().copy()

        triton_server_flags = self._get_triton_server_flags(models)
        self._server.update_config(params=triton_server_flags)

        rcg = RunConfigGeneratorFactory.create_run_config_generator(
            command_config=self._config,
            gpus=self._gpus,
            models=models,
            client=self._client,
            result_manager=self._result_manager,
            model_variant_name_manager=self._model_variant_name_manager)

        for run_config in rcg.get_configs():
            if self._state_manager.exiting():
                break

            if run_config.is_legal_combination():
                measurement = self._metrics_manager.execute_run_config(
                    run_config)

                self._check_for_valid_measurement(measurement)
                self._stop_ma_if_no_valid_measurement_threshold_reached()
            else:
                logger.info("Skipping illegal run configuration")
                measurement = None

            if measurement:
                objectives = [model.objectives() for model in models]
                weightings = [model.weighting() for model in models]

                measurement.set_metric_weightings(metric_objectives=objectives)
                measurement.set_constraint_manager(
                    constraint_manager=self._constraint_manager)
                measurement.set_model_config_weighting(
                    model_config_weights=weightings)

            rcg.set_last_results([measurement])
            self._state_manager.save_checkpoint()

        self._metrics_manager.finalize()

        # Reset the server args to global config
        self._server.update_config(params=server_config_copy.server_args())

        model_variant_name_manager_dict = self._state_manager.default_encode(
            self._model_variant_name_manager)

        self._state_manager.set_state_variable(
            'ModelManager.model_variant_name_manager',
            model_variant_name_manager_dict)

    def _get_triton_server_flags(self, models):
        triton_server_flags = models[0].triton_server_flags()

        for model in models:
            if model.triton_server_flags() != triton_server_flags:
                raise TritonModelAnalyzerException(
                    f"Triton server flags must be the same for all models to run concurrently"
                )

    def _check_for_ensemble_model_incompatibility(
            self, models: List[ConfigModelProfileSpec]) -> None:
        for model in models:
            model_config = ModelConfig.create_from_profile_spec(
                model, self._config, self._client, self._gpus)

            if model_config.is_ensemble():
                if len(models) > 1:
                    raise TritonModelAnalyzerException(
                        f'\nProfiling of multiple models is not supported for ensemble models'
                    )

                if self._config.run_config_search_mode == 'brute':
                    if self._config.get_config(
                    )['run_config_search_mode'].is_set_by_user():
                        raise TritonModelAnalyzerException(
                            f'\nBrute search mode is not supported for ensemble models'
                            '\nPlease use quick search mode (--run-config-search-mode quick)'
                        )
                    else:
                        self._config.run_config_search_mode = 'quick'
            elif not self._config.bls_composing_models:
                if len(self._config.cpu_only_composing_models) > 0:
                    raise TritonModelAnalyzerException(
                        f'\nCan only specify --cpu-only-composing-models for ensemble or BLS models.'
                    )

    def _init_state(self):
        """
        Sets ModelManager object managed
        state variables in AnalyzerState
        """

        self._state_manager.set_state_variable(
            'ModelManager.model_variant_name_manager',
            self._state_manager.default_encode(ModelVariantNameManager()))

    def _check_for_valid_measurement(
            self, measurement: Optional[RunConfigMeasurement]) -> None:
        if measurement:
            self._received_measurement_values_from_pa = True
        else:
            self._failed_measurement_attempts += 1

    def _stop_ma_if_no_valid_measurement_threshold_reached(self) -> None:
        if self._received_measurement_values_from_pa:
            return

        if self._failed_measurement_attempts >= INVALID_MEASUREMENT_THRESHOLD:
            raise TritonModelAnalyzerException(
                f'The first {INVALID_MEASUREMENT_THRESHOLD} attempts to acquire measurements ' \
                'have failed. Please examine the Tritonserver/PA error logs ' \
                'to determine what has gone wrong.'
            )
