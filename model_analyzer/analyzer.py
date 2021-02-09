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

from .config.full_run_config_generator import FullRunConfigGenerator
from .result.result_manager import ResultManager
from .record.metrics_manager import MetricsManager
from .output.file_writer import FileWriter
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

        # Results Manager
        self._result_manager = ResultManager(config=config)

        # Metrics Manager
        self._metrics_manager = MetricsManager(
            config=config,
            metric_tags=metric_tags,
            server=server,
            result_manager=self._result_manager)

    def run(self):
        """
        Configures RunConfigGenerator, then
        profiles for each run_config

        Raises
        ------
        TritonModelAnalyzerException
        """

        logging.info('Profiling server only metrics...')

        self._metrics_manager.profile_server(
            default_value=SERVER_ONLY_TABLE_DEFAULT_VALUE)

        for model_name in self._config.model_names:
            self._client.load_model(model_name=model_name)
            self._client.wait_for_model_ready(
                model_name=model_name, num_retries=self._config.max_retries)
            try:
                for run_config in FullRunConfigGenerator(
                        analyzer_config=self._config, model_name=model_name):

                    # Initialize the result
                    self._result_manager.init_result(run_config)

                    # TODO write/copy model configs
                    for perf_config in run_config.perf_analyzer_configs():
                        perf_output_writer = None if \
                            self._config.no_perf_output else FileWriter()

                        logging.info(
                            f"Profiling model {perf_config['model-name']}...")
                        self._metrics_manager.profile_model(
                            perf_config=perf_config,
                            perf_output_writer=perf_output_writer)

                    # Submit the result to be sorted
                    self._result_manager.complete_result()
            finally:
                self._client.unload_model(model_name=model_name)

            # Write results to tables
            self._result_manager.compile_results()

    def write_and_export_results(self):
        """
        Makes calls to _write_results out to streams or files. If
        exporting results is requested, uses a FileWriter for specified output
        files.
        """

        self._write_results(writer=FileWriter(), column_separator=' ')
        if self._config.export:
            server_metrics_path = os.path.join(
                self._config.export_path, self._config.filename_server_only)
            logging.info(
                f"Exporting server only metrics to {server_metrics_path}...")
            self._export_server_only_csv(
                writer=FileWriter(filename=server_metrics_path),
                column_separator=',')
            metrics_inference_path = os.path.join(
                self._config.export_path,
                self._config.filename_model_inference)
            metrics_gpu_path = os.path.join(self._config.export_path,
                                            self._config.filename_model_gpu)
            logging.info(
                f"Exporting inference metrics to {metrics_inference_path}...")
            logging.info(f"Exporting GPU metrics to {metrics_gpu_path}...")
            self._export_model_csv(
                inference_writer=FileWriter(filename=metrics_inference_path),
                gpu_metrics_writer=FileWriter(filename=metrics_gpu_path),
                column_separator=',')

    def _write_results(self, writer, column_separator):
        """
        Writes the tables using the writer with the given column
        specifications.

        Parameters
        ----------
        writer : OutputWriter
            Used to write the result tables to an output stream
        column_separator : str
            The string that will be inserted between each column
            of the table

        Raises
        ------
        TritonModelAnalyzerException
        """

        for table in self._result_manager.get_all_tables().values():
            self._write_result(table,
                               writer,
                               column_separator,
                               ignore_widths=False)

    def _write_result(self,
                      table,
                      writer,
                      column_separator,
                      ignore_widths=False,
                      include_title=True):
        """
        Utility function that writes any table
        """

        if include_title:
            writer.write('\n'.join([
                table.title() + ":",
                table.to_formatted_string(separator=column_separator,
                                          ignore_widths=ignore_widths), "\n"
            ]))
        else:
            writer.write(
                table.to_formatted_string(separator=column_separator,
                                          ignore_widths=ignore_widths) +
                "\n\n")

    def _export_server_only_csv(self, writer, column_separator):
        """
        Writes the server-only table as a csv file using the given writer

        Parameters
        ----------
        writer : OutputWriter
            Used to write the result tables to an output stream
        column_separator : str
            The string that will be inserted between each column
            of the table

        Raises
        ------
        TritonModelAnalyzerException
        """

        self._write_result(self._result_manager.get_server_table(),
                           writer,
                           column_separator,
                           ignore_widths=True,
                           include_title=False)

    def _export_model_csv(self, inference_writer, gpu_metrics_writer,
                          column_separator):
        """
        Writes the model table as a csv file using the given writer

        Parameters
        ----------
        inference_writer : OutputWriter
            Used to write the inference table result to an output stream
        gpu_metrics_writer : OutputWriter
            Used to write the gpu metrics table result to an output stream
        column_separator : str
            The string that will be inserted between each column
            of the table

        Raises
        ------
        TritonModelAnalyzerException
        """

        gpu_table, non_gpu_table = \
            self._result_manager.get_passing_model_tables()

        if non_gpu_table.empty() or gpu_table.empty():
            logging.info(
                "No results were found that satisfy specified constraints."
                "Writing results that failed constraints in sorted order.")
            gpu_table, non_gpu_table = \
                self._result_manager.get_failing_model_tables()

        self._write_result(table=gpu_table,
                           writer=gpu_metrics_writer,
                           column_separator=column_separator,
                           ignore_widths=True,
                           include_title=False)

        self._write_result(table=non_gpu_table,
                           writer=inference_writer,
                           column_separator=column_separator,
                           ignore_widths=True,
                           include_title=False)
