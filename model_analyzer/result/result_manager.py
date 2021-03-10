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

from .result_table import ResultTable
from .result_comparator import ResultComparator
from .model_result import ModelResult
from .measurement import Measurement
from model_analyzer.output.file_writer import FileWriter
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

import os
import heapq
import logging
from collections import defaultdict


class ResultManager:
    """
    This class provides methods to create, and add to
    ResultTables. Each ResultTable holds results from
    multiple runs.
    """

    non_gpu_specific_headers = [
        'Model', 'Batch', 'Concurrency', 'Model Config Path', 'Instance Group',
        'Dynamic Batcher Sizes', 'Satisfies Constraints'
    ]
    gpu_specific_headers = [
        'Model', 'GPU ID', 'Batch', 'Concurrency', 'Model Config Path',
        'Instance Group', 'Dynamic Batcher Sizes', 'Satisfies Constraints'
    ]
    server_table_headers = ['Model', 'GPU ID', 'Batch', 'Concurrency']
    server_only_table_key = 'server_gpu_metrics'
    model_gpu_table_key = 'model_gpu_metrics'
    model_inference_table_key = 'model_inference_metrics'

    def __init__(self, config, statistics):
        """
        Parameters
        ----------
        config : AnalyzerConfig
            the model analyzer config
        statistics: AnalyzerStatistics
            the statistics being collected for 
            this instance of model analyzer
        """

        self._config = config
        self._result_tables = {}
        self._current_run_result = None
        self._result_comparator = None
        self._statistics = statistics
        self._results = {}

        # Results are stored in a heap queue
        self._sorted_results = []
        self._passing_results = []
        self._failing_results = []

        # Results exported to export_path/results
        self._results_export_directory = os.path.join(config.export_path,
                                                      'results')
        os.makedirs(self._results_export_directory, exist_ok=True)

    def set_constraints_and_objectives(self, config_model):
        """
        Processes the constraints and objectives
        for given ConfigModel and creates a result
        comparator and constraint manager

        Parameters
        ----------
        config_model : ConfigModel
            The config model object for the model that is currently being
            run
        """

        self._constraints = config_model.constraints()
        self._result_comparator = ResultComparator(
            metric_objectives=config_model.objectives())

    def create_tables(self, gpu_specific_metrics, non_gpu_specific_metrics):
        """
        Creates the tables to print hold, display, and write
        results

        Parameters
        ----------
        gpu_specific_metrics : list of RecordTypes
            The metrics that have a GPU id associated with them
        non_gpu_specific_metrics : list of RecordTypes
            The metrics that do not have a GPU id associated with them
        """

        # Server only
        self._add_result_table(table_key=self.server_only_table_key,
                               title='Server Only',
                               headers=self.server_table_headers,
                               metric_types=gpu_specific_metrics)

        # Model Inference Tables
        self._add_result_table(table_key=self.model_gpu_table_key,
                               title='Models (GPU Metrics)',
                               headers=self.gpu_specific_headers,
                               metric_types=gpu_specific_metrics)

        self._add_result_table(table_key=self.model_inference_table_key,
                               title='Models (Inference)',
                               headers=self.non_gpu_specific_headers,
                               metric_types=non_gpu_specific_metrics)

    def init_result(self, run_config):
        """
        Initialize the ModelResults
        for the current model run.
        There will be one result per table.

        Parameters
        ----------
        run_config : RunConfig
            The run config corresponding to the current
            run.
        """

        # Create ModelResult
        self._current_run_result = ModelResult(
            run_config=run_config,
            comparator=self._result_comparator,
            constraints=self._constraints)

    def add_server_data(self, data, default_value):
        """
        Adds data to directly to the server only table

        Parameters
        ----------
        data : dict
            keys are gpu ids and values are lists of metric values
        default_value : val
            A value for those columns not applicable to standalone server
        """

        for gpu_id, metrics in data.items():
            data_row = ['triton-server', gpu_id, default_value, default_value]
            data_row += [metric.value() for metric in metrics]
            self._result_tables[
                self.server_only_table_key].insert_row_by_index(data_row)

    def add_measurement(self, run_config, measurement):
        """
        This function adds model inference
        measurements to the required result

        Parameters
        ----------
        run_config : RunConfig
            Contains the parameters used to generate the measurment
            like the model name, model_config_name
        measurement: Measurement
            the measurement to be added
        """

        if len(self._result_tables) == 0:
            raise TritonModelAnalyzerException(
                "Cannot add measurements without tables")
        elif not self._result_comparator:
            raise TritonModelAnalyzerException(
                "Cannot add measurements without setting result comparator")

        model_config_name = run_config.model_config().get_field('name')
        if model_config_name not in self._results:
            self._results[model_config_name] = ModelResult(
                model_name=run_config.model_name(),
                model_config=run_config.model_config(),
                comparator=self._result_comparator,
                constraints=self._constraints)
        measurement.set_result_comparator(comparator=self._result_comparator)
        self._results[model_config_name].add_measurement(measurement)

    def sort_results(self):
        """
        Sorts the results for all the model
        configs in the results structure,
        and puts them in the correct heap
        in descending order
        """

        for result in self._results.values():
            heapq.heappush(self._sorted_results, result)
            if result.failing():
                heapq.heappush(self._failing_results, result)
            else:
                heapq.heappush(self._passing_results, result)

    def top_n_results(self, n):
        """
        Parameters
        ----------
        n : int
            The number of  top results 
            to retrieve

        Returns
        -------
        list of ModelResults
            The n best results for this model,
            must all be passing results
        """

        if len(self._passing_results) == 0:
            logging.warn(
                f"Requested top {n} configs, but none satisfied constraints. "
                "Showing available constraint failing configs for this model.")

            if n > len(self._failing_results):
                logging.warn(
                    f"Requested top {n} failing configs, "
                    f"but found only {len(self._failing_results)}. "
                    "Showing all available constraint failing configs for this model."
                )
            return heapq.nsmallest(min(n, len(self._failing_results)),
                                   self._failing_results)

        if n > len(self._passing_results):
            logging.warn(
                f"Requested top {n} configs, "
                f"but found only {len(self._passing_results)} passing configs. "
                "Showing all available constraint satisfying configs for this model."
            )

        return heapq.nsmallest(min(n, len(self._passing_results)),
                               self._passing_results)

    def compile_results(self):
        """
        The function called at the end of all runs
        FOR A MODEL that compiles all result and
        dumps the data into tables for exporting.
        """

        # Fill rows in descending order
        while self._sorted_results:
            next_best_result = heapq.heappop(self._sorted_results)

            # Get name, instance_group, and dynamic batching enabled info from result
            model_name = next_best_result.model_name()
            instance_group_str = next_best_result.model_config(
            ).instance_group_string()
            dynamic_batching_str = next_best_result.model_config(
            ).dynamic_batching_string()
            self._compile_measurements(next_best_result, model_name,
                                       instance_group_str,
                                       dynamic_batching_str)

    def _compile_measurements(self, result, model_name, instance_group,
                              dynamic_batching):
        """
        checks measurement against constraints,
        and puts it into the correct (passing or failing)
        table
        """

        passing_measurements = result.passing_measurements()
        failing_measurements = result.failing_measurements()
        for (measurements, passes) in [(passing_measurements, True),
                                       (failing_measurements, False)]:
            while measurements:
                next_best_measurement = heapq.heappop(measurements)
                self._compile_measurement(model_name=model_name,
                                          instance_group=instance_group,
                                          dynamic_batching=dynamic_batching,
                                          measurement=next_best_measurement,
                                          passes=passes)

    def _compile_measurement(self, model_name, instance_group,
                             dynamic_batching, measurement, passes):
        """
        Add a single measurement to the specified
        table
        """

        perf_config = measurement.perf_config()
        tmp_model_name = perf_config['model-name']
        batch_size = perf_config['batch-size']
        concurrency = perf_config['concurrency-range']
        satisfies = "Yes" if passes else "No"

        # Non GPU specific data
        inference_metrics = [
            model_name, batch_size, concurrency, tmp_model_name,
            instance_group, dynamic_batching, satisfies
        ]
        inference_metrics += [
            round(metric.value(), 1) for metric in measurement.non_gpu_data()
        ]
        self._result_tables[
            self.model_inference_table_key].insert_row_by_index(
                row=inference_metrics)

        # GPU specific data
        for gpu_id, metrics in measurement.gpu_data().items():
            gpu_metrics = [
                model_name, gpu_id, batch_size, concurrency, tmp_model_name,
                instance_group, dynamic_batching, satisfies
            ]
            gpu_metrics += [round(metric.value(), 1) for metric in metrics]
            self._result_tables[self.model_gpu_table_key].insert_row_by_index(
                row=gpu_metrics)

    def _add_result_table(self, table_key, title, headers, metric_types):
        """
        Utility function that creates a table with column
        headers corresponding to perf_analyzer arguments
        and requested metrics. Also sets the result
        comparator for that table.
        """

        # Create headers
        table_headers = headers[:]
        for metric in metric_types:
            table_headers.append(metric.header())
        self._result_tables[table_key] = ResultTable(headers=table_headers,
                                                     title=title)

    def write_and_export_results(self):
        """
        Makes calls to _write_results out to streams or files. If
        exporting results is requested, uses a FileWriter for specified output
        files.
        """

        self._write_results(writer=FileWriter(), column_separator=' ')
        if self._config.export:

            # Configure server only results path and export results
            server_metrics_path = os.path.join(
                self._results_export_directory,
                self._config.filename_server_only)
            logging.info(
                f"Exporting server only metrics to {server_metrics_path}...")
            self._export_server_only_csv(
                writer=FileWriter(filename=server_metrics_path),
                column_separator=',')

            # Configure model metrics results path and export results
            metrics_inference_path = os.path.join(
                self._results_export_directory,
                self._config.filename_model_inference)
            metrics_gpu_path = os.path.join(self._results_export_directory,
                                            self._config.filename_model_gpu)
            logging.info(
                f"Exporting inference metrics to {metrics_inference_path}...")
            logging.info(f"Exporting GPU metrics to {metrics_gpu_path}...")
            self._export_model_csv(
                inference_writer=FileWriter(filename=metrics_inference_path),
                gpu_metrics_writer=FileWriter(filename=metrics_gpu_path),
                column_separator=',')

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

        self._write_result(self._result_tables[self.server_only_table_key],
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

        gpu_table = self._result_tables[self.model_gpu_table_key]
        non_gpu_table = self._result_tables[self.model_inference_table_key]

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

        for table in self._result_tables.values():
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

    def update_statistics(self, model_name):
        """
        This function computes statistics
        with results currently in the result
        manager's heap

        Parameters
        ----------
        model_name: str
            The name of the model whose statistics to
            update
        """

        passing_measurements = 0
        failing_measurements = 0
        total_configs = 0

        for result in self._sorted_results:
            total_configs += 1
            passing_measurements += len(result.passing_measurements())
            failing_measurements += len(result.failing_measurements())

        self._statistics.set_total_configurations(model_name, total_configs)
        self._statistics.set_passing_measurements(model_name,
                                                  passing_measurements)
        self._statistics.set_failing_measurements(model_name,
                                                  failing_measurements)
