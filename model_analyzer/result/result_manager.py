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
import heapq
import logging

from .result_table import ResultTable
from .result_comparator import ResultComparator
from .run_result import RunResult
from .measurement import Measurement
from .constraint_manager import ConstraintManager
from model_analyzer.output.file_writer import FileWriter
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class ResultManager:
    """
    This class provides methods to create, and add to
    ResultTables. Each ResultTable holds results from
    multiple runs.
    """

    non_gpu_specific_headers = [
        'Model', 'Batch', 'Concurrency', 'Model Config Path'
    ]
    gpu_specific_headers = [
        'Model', 'GPU ID', 'Batch', 'Concurrency', 'Model Config Path'
    ]
    server_table_headers = ['Model', 'GPU ID', 'Batch', 'Concurrency']
    server_only_table_key = 'server_gpu_metrics'
    model_gpu_table_passing_key = 'model_gpu_metrics_passing'
    model_inference_table_passing_key = 'model_inference_metrics_passing'
    model_gpu_table_failing_key = 'model_gpu_metrics_failing'
    model_inference_table_failing_key = 'model_inference_metrics_failing'

    def __init__(self, config):
        """
        Parameters
        ----------
        config : AnalyzerConfig
            the model analyzer config
        """

        self._config = config
        self._result_tables = {}
        self._current_run_result = None
        self._result_comparator = None

        # Results are stored in a heap queue
        self._results = []

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

        self._constraint_manager = ConstraintManager(
            constraints=config_model.constraints())
        self._result_comparator = ResultComparator(
            metric_objectives=config_model.objectives())

    def create_tables(self, gpu_specific_metrics, non_gpu_specific_metrics,
                      aggregation_tag):
        """
        Creates the tables to print hold, display, and write
        results

        Parameters
        ----------
        gpu_specific_metrics : list of RecordTypes
            The metrics that have a GPU id associated with them
        non_gpu_specific_metrics : list of RecordTypes
            The metrics that do not have a GPU id associated with them
        aggregation_tag : str
        """

        # Server only
        self._add_result_table(table_key=self.server_only_table_key,
                               title='Server Only',
                               headers=self.server_table_headers,
                               metric_types=gpu_specific_metrics,
                               aggregation_tag=aggregation_tag)

        # Model Inference Tables
        self._add_result_table(table_key=self.model_gpu_table_passing_key,
                               title='Models (GPU Metrics)',
                               headers=self.gpu_specific_headers,
                               metric_types=gpu_specific_metrics,
                               aggregation_tag=aggregation_tag)

        self._add_result_table(
            table_key=self.model_inference_table_passing_key,
            title='Models (Inference)',
            headers=self.non_gpu_specific_headers,
            metric_types=non_gpu_specific_metrics,
            aggregation_tag=aggregation_tag)

        self._add_result_table(
            table_key=self.model_gpu_table_failing_key,
            title='Models (GPU Metrics - Failed Constraints)',
            headers=self.gpu_specific_headers,
            metric_types=gpu_specific_metrics,
            aggregation_tag=aggregation_tag)

        self._add_result_table(
            table_key=self.model_inference_table_failing_key,
            title='Models (Inference - Failed Constraints)',
            headers=self.non_gpu_specific_headers,
            metric_types=non_gpu_specific_metrics,
            aggregation_tag=aggregation_tag)

    def init_result(self, run_config):
        """
        Initialize the RunResults
        for the current model run.
        There will be one result per table.

        Parameters
        ----------
        run_config : RunConfig
            The run config corresponding to the current
            run.
        """

        if len(self._result_tables) == 0:
            raise TritonModelAnalyzerException(
                "Cannot initialize results without tables")
        elif not self._result_comparator:
            raise TritonModelAnalyzerException(
                "Cannot initialize results without setting result comparator")

        # Create RunResult
        self._current_run_result = RunResult(
            run_config=run_config, comparator=self._result_comparator)

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

    def add_model_data(self, gpu_data, non_gpu_data, perf_config):
        """
        This function adds model inference
        measurements to the result, not directly
        to a table.

        Parameters
        ----------
        gpu_data : dict of list of Records
            These are the values from the monitors that have a GPU ID
            associated with them
        non_gpu_data : list of Records
            These do not have a GPU ID associated with them
        perf_config : PerfAnalyzerConfig
            The perf config that was used for the perf run that generated
            this data data
        """

        if self._current_run_result:
            measurement = Measurement(gpu_data=gpu_data,
                                      non_gpu_data=non_gpu_data,
                                      perf_config=perf_config,
                                      comparator=self._result_comparator)
            self._current_run_result.add_measurement(measurement)
            return measurement
        else:
            raise TritonModelAnalyzerException(
                "Must intialize a result before adding model data.")

    def complete_result(self):
        """
        Submit the current RunResults into
        the ResultTable
        """

        if self._current_run_result is not None:
            heapq.heappush(self._results, self._current_run_result)
            self._current_run_result = None

    def reset_result(self):
        """
        Submit the current RunResults into
        the ResultTable
        """

        self._current_run_result = None

    def top_n_results(self, n):
        """
        Parameters
        ----------
        n : int
            The number of  top results 
            to retrieve

        Returns
        -------
        RunResult
            The n best results for this model
        """

        return heapq.nsmallest(3, self._results)

    def compile_results(self):
        """
        The function called at the end of all runs
        FOR A MODEL that compiles all result and
        dumps the data into tables for exporting.
        """

        # Fill rows in descending order
        while self._results:
            next_best_result = heapq.heappop(self._results)
            model_name = next_best_result.run_config().model_name()
            measurements = next_best_result.measurements()
            self._compile_measurements(measurements, model_name)

    def _compile_measurements(self, measurements, model_name):
        """
        checks measurement against constraints,
        and puts it into the correct (passing or failing)
        table
        """

        while measurements:
            next_best_measurement = heapq.heappop(measurements)
            if self._constraint_manager.check_constraints(
                    measurement=next_best_measurement):
                self._compile_measurement(
                    model_name=model_name,
                    measurement=next_best_measurement,
                    gpu_table_key=self.model_gpu_table_passing_key,
                    inference_table_key=self.model_inference_table_passing_key)
            else:
                self._compile_measurement(
                    model_name=model_name,
                    measurement=next_best_measurement,
                    gpu_table_key=self.model_gpu_table_failing_key,
                    inference_table_key=self.model_inference_table_failing_key)

    def _compile_measurement(self, measurement, gpu_table_key,
                             inference_table_key, model_name):
        """
        Add a single measurement to the specified
        table
        """

        perf_config = measurement.perf_config()
        tmp_model_name = perf_config['model-name']
        batch_size = perf_config['batch-size']
        concurrency = perf_config['concurrency-range']

        # Non GPU specific data
        inference_metrics = [
            model_name, batch_size, concurrency, tmp_model_name
        ]
        inference_metrics += [
            metric.value() for metric in measurement.non_gpu_data()
        ]
        self._result_tables[inference_table_key].insert_row_by_index(
            row=inference_metrics)

        # GPU specific data
        for gpu_id, metrics in measurement.gpu_data().items():
            gpu_metrics = [
                model_name, gpu_id, batch_size, concurrency, tmp_model_name
            ]
            gpu_metrics += [metric.value() for metric in metrics]
            self._result_tables[gpu_table_key].insert_row_by_index(
                row=gpu_metrics)

    def _add_result_table(self,
                          table_key,
                          title,
                          headers,
                          metric_types,
                          aggregation_tag='Max'):
        """
        Utility function that creates a table with column
        headers corresponding to perf_analyzer arguments
        and requested metrics. Also sets the result
        comparator for that table.
        """

        # Create headers
        table_headers = headers[:]
        for metric in metric_types:
            table_headers.append(metric.header(aggregation_tag + " "))
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

        gpu_table = self._result_tables[self.model_gpu_table_passing_key]
        non_gpu_table = self._result_tables[
            self.model_inference_table_passing_key]

        if non_gpu_table.empty() or gpu_table.empty():
            logging.info(
                "No results were found that satisfy specified constraints."
                "Writing results that failed constraints in sorted order.")
            gpu_table = self._result_tables[self.model_gpu_table_failing_key]
            non_gpu_table = self._result_tables[
                self.model_inference_table_failing_key]

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
