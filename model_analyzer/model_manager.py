# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from model_analyzer.output.file_writer import FileWriter
from model_analyzer.result.measurement import Measurement
from model_analyzer.config.run.run_search import RunSearch
from model_analyzer.config.run.run_config_generator \
    import RunConfigGenerator
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

import logging
import os
import heapq
import shutil


class ModelManager:
    """
    This class handles the search for, creation of, and execution of run configs.
    It also records the best results for each model.
    """
    def __init__(self, config, client, server, metrics_manager,
                 result_manager):
        """
        Parameters
        ----------
        config: AnalyzerConfig
            The config for the model analyzer
        """

        self._config = config
        self._client = client
        self._server = server
        self._metrics_manager = metrics_manager
        self._result_manager = result_manager
        self._run_search = RunSearch(config=config)
        self._run_config_generator = RunConfigGenerator(config=config,
                                                        client=self._client)

        # Generate the output model repository path folder.
        self._output_model_repo_path = config.output_model_repository_path
        try:
            os.mkdir(self._output_model_repo_path)
        except OSError:
            if not config.override_output_model_repository:
                raise TritonModelAnalyzerException(
                    f'Path "{self._output_model_repo_path}" already exists. '
                    'Please set or modify "--output-model-repository-path" flag or remove this directory.'
                    ' You can also allow overriding of the output directory using'
                    ' the "--override-output-model-repository" flag.')
            else:
                shutil.rmtree(self._output_model_repo_path)
                logging.warn(
                    f'Overriding the output model repo path "{self._output_model_repo_path}"...'
                )
                os.mkdir(self._output_model_repo_path)

    def run_model(self, model):
        """
        Generates configs, runs inferences, gets
        measurements for a single model

        Parameters
        ----------
        model : ConfigModel
            The model being run
        """

        # Clear any configs from previous model run
        self._run_config_generator.clear_configs()

        # Update the server's config for this model run
        self._server.update_config(params=model.triton_server_flags())

        if self._config.run_config_search_disable:
            self._run_model_no_search(model)
        else:
            self._run_model_with_search(model)

        # Sort the results for this model
        self._result_manager.sort_results(model_name=model.model_name())

    def _run_model_no_search(self, model):
        """
        Creates run configs from specified combinations and executes
        them without any run search
        """

        # Generate all the run configs at once and return
        if self._config.triton_launch_mode != 'remote':
            user_model_config_sweeps = \
                self._run_config_generator.generate_model_config_combinations(
                    model.model_config_parameters())
            for user_model_config_sweep in user_model_config_sweeps:
                self._run_config_generator.generate_run_config_for_model_sweep(
                    model, user_model_config_sweep)
        else:
            self._run_config_generator.generate_run_config_for_model_sweep(
                model, None)
        self._execute_run_configs()

    def _run_model_with_search(self, model):
        """
        Searches over the required config elements,
        creates run configs and executes them
        """

        model_config_parameters = model.model_config_parameters()

        # Run config search is enabled, figure out which parameters to sweep over and do sweep
        if self._config.triton_launch_mode == 'remote':
            self._run_model_config_sweep(model, search_model_config=False)
        else:
            if model_config_parameters:
                user_model_config_sweeps = \
                    self._run_config_generator.generate_model_config_combinations(
                        model_config_parameters)
                if model.parameters()['concurrency']:
                    # Both are specified, search over neither
                    for user_model_config_sweep in user_model_config_sweeps:
                        self._run_config_generator.generate_run_config_for_model_sweep(
                            model, user_model_config_sweep)
                    self._execute_run_configs()
                else:
                    # Search through concurrency values only
                    for user_model_config_sweep in user_model_config_sweeps:
                        self._run_model_config_sweep(
                            model,
                            search_model_config=False,
                            user_model_config_sweep=user_model_config_sweep)
            else:
                # Model Config parameters unspecified
                self._run_model_config_sweep(model, search_model_config=True)

    def _run_model_config_sweep(self,
                                model,
                                search_model_config,
                                user_model_config_sweep=None):
        """
        Initializes the model sweep, iterates until search bounds,
        and executes run configs 
        """

        self._run_search.init_model_sweep(model.parameters()['concurrency'],
                                          search_model_config)

        next_model = model
        while True:
            next_model, auto_model_config_sweep = self._run_search.get_model_sweep(
                next_model)

            # End search when get_model sweep returns empty
            if not auto_model_config_sweep:
                break
            if user_model_config_sweep:
                model_sweep_for_run_config = user_model_config_sweep
            else:
                model_sweep_for_run_config = auto_model_config_sweep[0]

            self._run_config_generator.generate_run_config_for_model_sweep(
                next_model, model_sweep_for_run_config)
            self._run_search.add_measurements(self._execute_run_configs())

    def _execute_run_configs(self):
        """
        Executes the run configs stored in the run
        config generator until there are none left.
        Returns obtained measurements. Also sends them
        to the result manager
        """

        measurements = []
        while self._run_config_generator.run_configs():
            # Remove one run config from the list
            run_config = self._run_config_generator.next_config()

            # Start server, and load model variant
            self._server.start()
            if not self._create_and_load_model_variant(
                    original_name=run_config.model_name(),
                    variant_config=run_config.model_config()):
                continue

            # Profile various batch size and concurrency values.
            # TODO: Need to sort the values for batch size and concurrency
            # for correct measurment of the GPU memory metrics.
            perf_output_writer = None if \
                not self._config.perf_output else FileWriter()
            perf_config = run_config.perf_config()

            logging.info(f"Profiling model {perf_config['model-name']}...")
            gpu_data, non_gpu_data = self._metrics_manager.profile_model(
                perf_config=perf_config, perf_output_writer=perf_output_writer)
            if gpu_data is not None and non_gpu_data is not None:
                measurement = Measurement(gpu_data=gpu_data,
                                          non_gpu_data=non_gpu_data,
                                          perf_config=perf_config)
                self._result_manager.add_measurement(run_config, measurement)
                measurements.append(measurement)

            self._server.stop()
            if self._config.triton_output_path:
                self._server.write_server_logs(self._config.triton_output_path)
        return measurements

    def _create_and_load_model_variant(self, original_name, variant_config):
        """
        Creates a directory for the model config
        variant in the output model repository
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
                variant_config.write_config_to_file(new_model_dir, True,
                                                    original_model_dir)
            except FileExistsError:
                pass

        self._client.wait_for_server_ready(self._config.max_retries)

        if self._client.load_model(model_name=variant_name) == -1:
            return False

        if self._client.wait_for_model_ready(
                model_name=variant_name,
                num_retries=self._config.max_retries) == -1:
            return False
        return True
