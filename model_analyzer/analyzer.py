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

from model_analyzer.model_manager import ModelManager
from .analyzer_statistics import AnalyzerStatistics
from .result.result_manager import ResultManager
from .record.metrics_manager import MetricsManager
from .plots.plot_manager import PlotManager
from .reports.report_manager import ReportManager

import logging
import os


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

        self._statistics = AnalyzerStatistics(config=config)

        self._result_manager = ResultManager(config=config,
                                             statistics=self._statistics)

        self._metrics_manager = MetricsManager(
            config=config,
            metric_tags=metric_tags,
            server=server,
            result_manager=self._result_manager)

        self._model_manager = ModelManager(
            config=config,
            client=client,
            server=server,
            result_manager=self._result_manager,
            metrics_manager=self._metrics_manager,
        )

        self._plot_manager = PlotManager(config=config)

        self._report_manager = ReportManager(config=config)

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
        self._metrics_manager.profile_server()
        self._server.stop()

        # Phase 2: Profile each model
        for model in config.model_names:
            self._result_manager.set_constraints_and_objectives(
                config_model=model)

            self._model_manager.run_model(model=model)

            if self._config.summarize:
                # Send requested best results to plot manager
                self._process_top_results()
                # Plot data to graphs
                self._plot_manager.compile_plots(model_name=model.model_name())

            # Dump results to tables
            self._result_manager.update_statistics(model.model_name())
            self._result_manager.compile_results()

        # If requested, save top n models
        self._save_top_models()

    def write_and_export_results(self):
        """
        Tells the results manager and plot managers
        to dump the results onto disk
        """

        self._result_manager.write_and_export_results()
        if self._config.summarize:
            self._plot_manager.export_plots()
            self._report_manager.export_summary(statistics=self._statistics)

    def _process_top_results(self):
        """
        Add the best measurements from the 
        best result to the plot for the model.
        This function must be called before 
        corresponding plots are completed,
        and results are
        """

        for result in self._result_manager.top_n_results(
                n=self._config.top_n_configs):

            # Send all measurements to plots manager
            self._plot_manager.add_result(result=result)

            # Send top_n measurements to report manager
            self._report_manager.add_result(result=result)

    def _save_top_models(self):
        """
        If requested, save the top models to a directory in
        the export path
        """

        # Create top model directory
        top_model_export_directory = os.path.join(self._config.export_path,
                                                  'best_models')
        os.makedirs(top_model_export_directory, exist_ok=True)

        for result, model_name in self._model_manager.top_n_models(
                n=self._config.top_n_models):

            # Create model directory for best model
            next_model_dir = os.path.join(top_model_export_directory,
                                          model_name)
            os.makedirs(next_model_dir, exist_ok=True)

            # Ensure model config name is correct, and write
            next_model_config = result.model_config()
            next_model_config.set_field('name', model_name)

            original_model_dir = os.path.join(self._config.model_repository,
                                              model_name)
            next_model_config.write_config_to_file(next_model_dir, True,
                                                   original_model_dir)
