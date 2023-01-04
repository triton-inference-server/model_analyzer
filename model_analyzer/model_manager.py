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

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.config.generate.run_config_generator_factory import RunConfigGeneratorFactory
from .model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.config.generate.model_variant_name_manager import ModelVariantNameManager
from model_analyzer.result.constraint_manager import ConstraintManager

import logging

logger = logging.getLogger(LOGGER_NAME)


class ModelManager:
    """
    This class handles the search for, creation of, and execution of run configs.
    It also records the best results for each model.
    """

    def __init__(self, config, gpus, client, server, metrics_manager,
                 result_manager, state_manager):
        """
        Parameters
        ----------
        config:ConfigCommandProfile
            The config for the model analyzer
        gpus: List of GPUDevices
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
        """

        self._config = config
        self._gpus = gpus
        self._client = client
        self._server = server
        self._metrics_manager = metrics_manager
        self._result_manager = result_manager
        self._state_manager = state_manager

        if state_manager.starting_fresh_run():
            self._init_state()

        self._model_variant_name_manager = ModelVariantNameManager.from_dict(
            self._state_manager.get_state_variable(
                'ModelManager.model_variant_name_manager'))

    def run_models(self, models):
        """
        Generates configs, runs inferences, gets
        measurements for a list of models

        Parameters
        ----------
        models : List of ConfigModelProfileSpec
            The models to run
        """

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
            else:
                logger.info("Skipping illegal run configuration")
                measurement = None

            if measurement:
                objectives = [model.objectives() for model in models]
                constraints = [model.constraints() for model in models]
                weightings = [model.weighting() for model in models]

                measurement.set_metric_weightings(metric_objectives=objectives)
                measurement.set_model_config_constraints(
                    model_config_constraints=constraints)
                measurement.set_model_config_weighting(
                    model_config_weights=weightings)

            rcg.set_last_results([measurement])

        self._metrics_manager.finalize()

        # Reset the server args to global config
        self._server.update_config(params=server_config_copy.server_args())

        model_variant_name_manager_dict = self._state_manager.default_encode(
            rcg._model_variant_name_manager)

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

    def _init_state(self):
        """
        Sets ModelManager object managed
        state variables in AnalyerState
        """

        self._state_manager.set_state_variable(
            'ModelManager.model_variant_name_manager',
            self._state_manager.default_encode(ModelVariantNameManager()))
