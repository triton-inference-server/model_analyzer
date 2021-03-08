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

import os
import logging

from .output.file_writer import FileWriter
from .config.run.run_config_generator import RunConfigGenerator
from .result.result_manager import ResultManager
from .record.metrics_manager import MetricsManager
from .plots.plot_manager import PlotManager
from .constants import SERVER_ONLY_TABLE_DEFAULT_VALUE


class Analyzer:
    """
    A class responsible for coordinating the various components of the
    model_analyzer. Configured with metrics to monitor, exposes profiling and
    result writing methods.
    """

    def __init__(self, config, client, metric_tags, server):
        """
        Parameters
        ----------
        config : Config
            Model Analyzer config
        client : TritonClient
            Instance used to load/unload models
        metric_tags : List of str
            The list of metric tags corresponding to the metrics to monitor.
        server : TritonServer handle
        """

        self._config = config
        self._client = client
        self._server = server

        # Results Manager
        self._result_manager = ResultManager(config=config)

        # Metrics Manager
        self._metrics_manager = MetricsManager(
            config=config,
            metric_tags=metric_tags,
            server=server,
            result_manager=self._result_manager)

        # Plot manager
        self._plot_manager = PlotManager(config=config)

    def run(self):
        """
        Configures RunConfigGenerator, then
        profiles for each run_config

        Raises
        ------
        TritonModelAnalyzerException
        """

        config = self._config

        logging.info('Profiling server only metrics...')

        # Phase 1: Profile server only metrics
        self._server.start()
        self._client.wait_for_server_ready(config.max_retries)
        self._metrics_manager.profile_server(
            default_value=SERVER_ONLY_TABLE_DEFAULT_VALUE)
        self._server.stop()

        output_model_repo_path = config.output_model_repository_path
        model_repository = config.model_repository

        # Phase 2: Profile each model
        for model in config.model_names:
            self._result_manager.set_constraints_and_objectives(
                config_model=model)

            run_config_generator = RunConfigGenerator(
                model, client=self._client, analyzer_config=self._config)
            for run_config in run_config_generator.get_run_configs():
                model_config = run_config.model_config()
                original_model_name = run_config.model_name()
                model_name = model_config.get_field('name')

                # Create the directory for the new model
                os.mkdir(f'{output_model_repo_path}/{model_name}')
                model_config.write_config_to_file(
                    f'{output_model_repo_path}/{model_name}', True,
                    f'{model_repository}/{original_model_name}')

                self._server.start()
                self._client.wait_for_server_ready(config.max_retries)
                self._client.load_model(model_name=model_name)
                self._client.wait_for_model_ready(
                    model_name=model_name,
                    num_retries=self._config.max_retries)
                self._result_manager.init_result(run_config)

                # Profile various batch size and concurrency values.
                # TODO: Need to sort the values for batch size and concurrency
                # for correct measurment of the GPU memory metrics.
                for perf_config in run_config.perf_analyzer_configs():
                    perf_output_writer = None if \
                        not self._config.perf_output else FileWriter()

                    logging.info(
                        f"Profiling model {perf_config['model-name']}...")
                    self._metrics_manager.profile_model(
                        perf_config=perf_config,
                        perf_output_writer=perf_output_writer)
                self._server.stop()

                # Submit the result to be sorted
                self._result_manager.complete_result()

            # Add the best measurements from the best result to the plot for the model
            for result in self._result_manager.top_n_results(
                    n=config.top_n_configs):
                model_config_name = result.run_config().model_config(
                ).get_field('name')
                for measurement in result.top_n_measurements(
                        n=config.top_n_measurements):
                    self._plot_manager.add_measurement(
                        model_config_label=model_config_name,
                        measurement=measurement)

            # Write plots to disk
            self._plot_manager.complete_plots(model_name=model.model_name())

            # Write results to tables
            self._result_manager.compile_results()

    def write_and_export_results(self):
        """
        Tells the results manager and plot managers
        to dump the results onto disk
        """

        self._result_manager.write_and_export_results()
        self._plot_manager.export_plots()
