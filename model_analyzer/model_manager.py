# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.output.file_writer import FileWriter
from model_analyzer.config.run.run_config_generator \
    import RunConfigGenerator

import os
import logging

logger = logging.getLogger(LOGGER_NAME)


class ModelManager:
    """
    This class handles the search for, creation of, and execution of run configs.
    It also records the best results for each model.
    """

    def __init__(self, config, client, server, metrics_manager, result_manager,
                 state_manager):
        """
        Parameters
        ----------
        config:ConfigCommandProfile
            The config for the model analyzer
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
        self._client = client
        self._server = server
        self._metrics_manager = metrics_manager
        self._result_manager = result_manager
        self._state_manager = state_manager
        self._first_config_variant = None

        # Generate the output model repository path folder.
        self._output_model_repo_path = config.output_model_repository_path

    def run_model(self, model):
        """
        Generates configs, runs inferences, gets
        measurements for a single model

        Parameters
        ----------
        model : ConfigModelProfileSpec
            The model being run
        """

        # Reset first config variant for new model
        self._first_config_variant = None

        # Save the global server config and update the server's config for this model run
        server_config_copy = self._server.config().copy()
        self._server.update_config(params=model.triton_server_flags())

        rcg = RunConfigGenerator(config=self._config, client=self._client)
        rcg.init(model)
        while not rcg.is_done() and not self._state_manager.exiting():
            run_config = rcg.next_config()
            measurement = self._execute_run_config(run_config)
            rcg.add_measurement(measurement)

        # Reset the server args to global config
        self._server.update_config(params=server_config_copy.server_args())

    def _execute_run_config(self, run_config):
        """
        Executes the run config. Returns obtained measurement. Also sends 
        measurement to the result manager
        """

        # Create model variant
        self._create_model_variant(original_name=run_config.model_name(),
                                   variant_config=run_config.model_config())

        # If this run config was already run, do not run again, just get the measurement
        measurement = self._get_measurement_if_config_duplicate(run_config)
        if measurement:
            return measurement

        # Start server, and load model variant
        self._server.start(env=run_config.triton_environment())
        if not self._load_model_variant(
                variant_config=run_config.model_config()):
            self._server.stop()
            return

        # Profile various batch size and concurrency values.
        # TODO: Need to sort the values for batch size and concurrency
        # for correct measurment of the GPU memory metrics.
        perf_output_writer = None if \
            not self._config.perf_output else FileWriter(self._config.perf_output_path)
        perf_config = run_config.perf_config()

        logger.info(f"Profiling model {perf_config['model-name']}...")
        measurement = self._metrics_manager.profile_model(
            run_config=run_config, perf_output_writer=perf_output_writer)
        self._server.stop()

        return measurement

    def _create_model_variant(self, original_name, variant_config):
        """
        Creates a directory for the model config variant in the output model
        repository and fills directory with config
        """

        variant_name = variant_config.get_field('name')
        if self._config.triton_launch_mode != 'remote':
            model_repository = self._config.model_repository

            original_model_dir = os.path.join(model_repository, original_name)
            new_model_dir = os.path.join(self._output_model_repo_path,
                                         variant_name)
            try:
                # Create the directory for the new model
                os.makedirs(new_model_dir, exist_ok=False)
                variant_config.write_config_to_file(new_model_dir,
                                                    original_model_dir,
                                                    self._first_config_variant)
                if self._first_config_variant is None:
                    self._first_config_variant = os.path.join(
                        self._output_model_repo_path, variant_name)
            except FileExistsError:
                pass

    def _load_model_variant(self, variant_config):
        """
        Loads a model variant in the client
        """

        variant_name = variant_config.get_field('name')
        if self._config.triton_launch_mode != 'c_api':
            self._client.wait_for_server_ready(self._config.client_max_retries)

            if self._client.load_model(model_name=variant_name) == -1:
                return False

            if self._client.wait_for_model_ready(
                    model_name=variant_name,
                    num_retries=self._config.client_max_retries) == -1:
                return False
        return True

    def _get_measurement_if_config_duplicate(self, run_config):
        """
        Checks whether this run config has measurements
        in the state manager's results object
        """

        model_name = run_config.model_name()
        model_config_name = run_config.model_config().get_field('name')
        perf_config_str = run_config.perf_config().representation()

        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        # check whether perf config string is a key in result dict
        if model_name not in results:
            return False
        if not self._is_config_in_results(
                run_config.model_config()._model_config, results[model_name]):
            return False
        measurements = results[model_name][model_config_name][1]

        # For backward compatibility with keys that still have -u,
        # we will remove -u from all keys, convert to set and check
        # perf_config_str is present
        if perf_config_str in set(
                map(PerfAnalyzerConfig.remove_url_from_cli_string,
                    measurements.keys())):
            return measurements[perf_config_str]
        else:
            return None

    def _is_config_in_results(self, config, model_results):
        """
        Returns true if `config` exists in the checkpoint `model_results`
        """

        for result in model_results.values():
            if config == result[0]._model_config:
                return True
        return False
