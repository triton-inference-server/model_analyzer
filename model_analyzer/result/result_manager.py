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
from model_analyzer.constants import TOP_MODELS_REPORT_KEY
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from .result_heap import ResultHeap
from .result_table import ResultTable
from .result_comparator import ResultComparator
from .model_result import ModelResult

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

    headers = {
        'model_name': 'Model',
        'batch_size': 'Batch',
        'concurrency': 'Concurrency',
        'model_config_path': 'Model Config Path',
        'instance_group': 'Instance Group',
        'dynamic_batch_sizes': 'Preferred Batch Sizes',
        'satisfies_constraints': 'Satisfies Constraints',
        'gpu_id': 'GPU ID'
    }

    server_only_table_key = 'server_gpu_metrics'
    model_gpu_table_key = 'model_gpu_metrics'
    model_inference_table_key = 'model_inference_metrics'

    def __init__(self, config, statistics, state_manager):
        """
        Parameters
        ----------
        config : AnalyzerConfig
            the model analyzer config
        statistics: AnalyzerStatistics
            the statistics being collected for this instance of model
            analyzer
        state_manager: AnalyzerStateManager
            The object that allows control and update of state
        """

        self._config = config
        self._is_cpu_only = config.cpu_only
        self._result_tables = {}
        self._statistics = statistics
        self._state_manager = state_manager

        if state_manager.starting_fresh_run():
            self._init_state()

        # Data structures for sorting results
        self._per_model_sorted_results = defaultdict(ResultHeap)
        self._across_model_sorted_results = ResultHeap()

        # Headers Dictionary and result tables
        self._gpu_metrics_to_headers = {}
        self._non_gpu_metrics_to_headers = {}
        self._result_tables = {}

        # Results exported to export_path/results
        self._results_export_directory = os.path.join(config.export_path,
                                                      'results')
        os.makedirs(self._results_export_directory, exist_ok=True)

    def _init_state(self):
        """
        Sets ResultManager object managed
        state variables in AnalyerState
        """

        self._state_manager.set_state_variable('ResultManager.results', {})

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

        for metric in gpu_specific_metrics:
            self._gpu_metrics_to_headers[metric.tag] = metric.header()

        for metric in non_gpu_specific_metrics:
            self._non_gpu_metrics_to_headers[metric.tag] = metric.header()

    def _create_server_table(self):
        # Server only
        server_output_headers = []
        server_output_fields = self._config.server_output_fields
        for server_output_field in server_output_fields:
            if server_output_field in self.headers:
                server_output_headers.append(self.headers[server_output_field])
            elif server_output_field in self._gpu_metrics_to_headers:
                server_output_headers.append(
                    self._gpu_metrics_to_headers[server_output_field])
            else:
                raise TritonModelAnalyzerException(
                    f'Server output field "{server_output_field}", does not exist'
                )
        self._add_result_table(table_key=self.server_only_table_key,
                               title='Server Only',
                               headers=server_output_headers)
        self._server_output_fields = server_output_fields
    
    def _create_inference_table(self):
        # Inference only
        inference_output_headers = []
        inference_output_fields = self._config.inference_output_fields
        for inference_output_field in inference_output_fields:
            if inference_output_field in self.headers:
                inference_output_headers.append(
                    self.headers[inference_output_field])
            elif inference_output_field in self._non_gpu_metrics_to_headers:
                inference_output_headers.append(
                    self._non_gpu_metrics_to_headers[inference_output_field])
            else:
                raise TritonModelAnalyzerException(
                    f'Inference output field "{inference_output_field}", does not exist'
                )
        self._inference_output_fields = inference_output_fields

        self._add_result_table(
            table_key=self.model_inference_table_key,
            title='Models (Inference)',
            headers=inference_output_headers,
        )
    
    def _create_gpu_table(self):
        gpu_output_headers = []
        gpu_output_fields = self._config.gpu_output_fields
        for gpu_output_field in gpu_output_fields:
            if gpu_output_field in self.headers:
                gpu_output_headers.append(self.headers[gpu_output_field])
            elif gpu_output_field in self._gpu_metrics_to_headers:
                gpu_output_headers.append(
                    self._gpu_metrics_to_headers[gpu_output_field])
            else:
                raise TritonModelAnalyzerException(
                    f'GPU output field "{gpu_output_field}", does not exist')
        self._gpu_output_fields = gpu_output_fields

        # Model GPU Metrics
        self._add_result_table(table_key=self.model_gpu_table_key,
                               title='Models (GPU Metrics)',
                               headers=gpu_output_headers)

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
        for metric in gpu_specific_metrics:
            self._gpu_metrics_to_headers[metric.tag] = metric.header()

        for metric in non_gpu_specific_metrics:
            self._non_gpu_metrics_to_headers[metric.tag] = metric.header()
        
        self._create_inference_table()

        if not self._is_cpu_only:
            self._create_gpu_table()
            self._create_server_table()
        else:
            logging.info('No GPU detected, will only export inference results.')

    def _find_index_for_field(self, fields, field_name):
        try:
            index = fields.index(field_name)
            return index
        except ValueError:
            return None

    def add_server_data(self, data):
        """
        Adds data to directly to the server only table

        Parameters
        ----------
        data : dict
            keys are gpu ids and values are lists of metric values
        """

        server_fields = self._server_output_fields

        for gpu_id, metrics in data.items():
            data_row = [None] * len(server_fields)

            model_name_index = self._find_index_for_field(
                server_fields, 'model_name')
            if model_name_index is not None:
                data_row[model_name_index] = 'triton-server'

            gpu_id_index = self._find_index_for_field(server_fields, 'gpu_id')
            if gpu_id_index is not None:
                data_row[gpu_id_index] = gpu_id

            for metric in metrics:
                metric_tag_index = self._find_index_for_field(
                    server_fields, metric.tag)

                if metric_tag_index is not None:
                    data_row[metric_tag_index] = round(metric.value(), 1)
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

        model_name = run_config.model_name()
        model_config = run_config.model_config()
        model_config_name = model_config.get_field('name')

        # Get reference to results state and modify it
        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        if model_name not in results:
            results[model_name] = {}
        if model_config_name not in results[model_name]:
            results[model_name][model_config_name] = (model_config, {})

        measurement_key = measurement.perf_config().to_cli_string()
        results[model_name][model_config_name][1][
            measurement_key] = measurement

    def collect_and_sort_results(self, num_models):
        """
        Collects objectives and constraints for
        each model, constructs results from the
        measurements obtained, and sorts and 
        filters them according to constraints
        and objectives.

        Parameters
        ----------
        num_models: int
            The number of config models to be included in the
            results
        """

        # Collect objectives and constraints
        comparators = {}
        constraints = {}
        for model in self._config.model_names:
            comparators[model.model_name()] = ResultComparator(
                metric_objectives=model.objectives())
            constraints[model.model_name()] = model.constraints()

        # Construct and add results to individual result heaps as well as global result heap
        results = self._state_manager.get_state_variable(
            'ResultManager.results')
        for model_name in [
                model.model_name()
                for model in self._config.model_names[:num_models]
        ]:
            if model_name not in results:
                logging.warn(
                    f"Model {model_name} requested for analysis but no results were found. "
                    "Ensure that this model was actually profiled.")
            else:
                result_dict = results[model_name]
                for (model_config, measurements) in result_dict.values():
                    result = ModelResult(model_name=model_name,
                                         model_config=model_config,
                                         comparator=comparators[model_name],
                                         constraints=constraints[model_name])
                    for measurement in measurements.values():
                        measurement.set_result_comparator(
                            comparator=comparators[model_name])
                        result.add_measurement(measurement)
                    self._per_model_sorted_results[model_name].add_result(
                        result)
                    self._across_model_sorted_results.add_result(result)

    def top_n_results(self, model_name=None, n=-1):
        """
        Parameters
        ----------
        model_name: str
            The name of the model
            for which we need the top 
            n results.
        n : int
            The number of  top results
            to retrieve. Returns all by 
            default

        Returns
        -------
        list of ModelResults
            The n best results for this model,
            must all be passing results
        """

        if model_name:
            result_heap = self._per_model_sorted_results[model_name]
        else:
            result_heap = self._across_model_sorted_results
        return result_heap.top_n_results(n)

    def tabulate_results(self):
        """
        The function called at the end of all runs
        FOR ALL MODELs that compiles all results and
        dumps the data into tables for exporting.
        """

        self._update_statistics()

        # Fill rows in descending order
        for result_heap in self._per_model_sorted_results.values():
            while not result_heap.empty():
                self._tabulate_measurements(result_heap.next_best_result())

    def _tabulate_measurements(self, result):
        """
        checks measurement against constraints,
        and puts it into the correct (passing or failing)
        table
        """

        model_name = result.model_name()
        instance_group = result.model_config().instance_group_string()
        dynamic_batching = result.model_config().dynamic_batching_string()

        passing_measurements = result.passing_measurements()
        failing_measurements = result.failing_measurements()

        for (measurements, passes) in [(passing_measurements, True),
                                       (failing_measurements, False)]:
            while measurements:
                next_best_measurement = heapq.heappop(measurements)
                self._tabulate_measurement(model_name=model_name,
                                           instance_group=instance_group,
                                           dynamic_batching=dynamic_batching,
                                           measurement=next_best_measurement,
                                           passes=passes)

    def _get_common_row_items(self, fields, batch_size, concurrency, satisfies,
                              model_name, model_config_path, dynamic_batching,
                              instance_group):
        row = [None] * len(fields)

        # Model Name
        model_name_index = self._find_index_for_field(fields, 'model_name')
        if model_name_index is not None:
            row[model_name_index] = model_name

        # Batch Size
        batch_size_index = self._find_index_for_field(fields, 'batch_size')
        if batch_size_index is not None:
            row[batch_size_index] = batch_size

        # Concurrency
        concurrency_index = self._find_index_for_field(fields, 'concurrency')
        if concurrency_index is not None:
            row[concurrency_index] = concurrency

        # Satisfies
        satisfies_constraints_index = self._find_index_for_field(
            fields, 'satisfies_constraints')
        if satisfies_constraints_index is not None:
            row[satisfies_constraints_index] = satisfies

        # Model Config Path
        model_config_path_idx = self._find_index_for_field(
            fields, 'model_config_path')
        if model_config_path_idx is not None:
            row[model_config_path_idx] = model_config_path

        # Dynamic Batching
        dynamic_batching_idx = self._find_index_for_field(
            fields, 'dynamic_batch_sizes')
        if dynamic_batching_idx is not None:
            row[dynamic_batching_idx] = dynamic_batching

        # Instance Group
        instance_group_idx = self._find_index_for_field(
            fields, 'instance_group')
        if instance_group_idx is not None:
            row[instance_group_idx] = instance_group
        return row

    def _tabulate_measurement(self, model_name, instance_group,
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
        inference_fields = self._inference_output_fields
        inference_row = self._get_common_row_items(
            inference_fields, batch_size, concurrency, satisfies, model_name,
            tmp_model_name, dynamic_batching, instance_group)

        for metric in measurement.non_gpu_data():
            metric_tag_index = self._find_index_for_field(
                inference_fields, metric.tag)

            if metric_tag_index is not None:
                inference_row[metric_tag_index] = round(metric.value(), 1)

        self._result_tables[
            self.model_inference_table_key].insert_row_by_index(inference_row)

        if not self._is_cpu_only:
            # GPU specific data
            for gpu_id, metrics in measurement.gpu_data().items():
                gpu_fields = self._gpu_output_fields
                gpu_row = self._get_common_row_items(gpu_fields, batch_size,
                                                     concurrency, satisfies,
                                                     model_name, tmp_model_name,
                                                     dynamic_batching,
                                                     instance_group)
                gpu_id_index = self._find_index_for_field(gpu_fields, 'gpu_id')
                if gpu_id_index is not None:
                    gpu_row[gpu_id_index] = gpu_id
                for metric in metrics:
                    metric_tag_index = self._find_index_for_field(
                        gpu_fields, metric.tag)
                    if metric_tag_index is not None:
                        gpu_row[metric_tag_index] = round(metric.value(), 1)
                self._result_tables[self.model_gpu_table_key].insert_row_by_index(
                    row=gpu_row)

    def _add_result_table(self, table_key, title, headers):
        """
        Utility function that creates a table with column
        headers corresponding to perf_analyzer arguments
        and requested metrics. Also sets the result
        comparator for that table.
        """

        # Create headers
        self._result_tables[table_key] = ResultTable(headers=headers,
                                                     title=title)

    def write_and_export_results(self):
        """
        Makes calls to _write_results out to streams or files. If
        exporting results is requested, uses a FileWriter for specified output
        files.
        """

        self._write_results(writer=FileWriter(), column_separator=' ')
        if self._config.export:
            if not self._is_cpu_only:
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

        if not self._is_cpu_only:
            gpu_table = self._result_tables[self.model_gpu_table_key]
            self._write_result(table=gpu_table,
                               writer=gpu_metrics_writer,
                               column_separator=column_separator,
                               ignore_widths=True,
                               include_title=False)

        non_gpu_table = self._result_tables[self.model_inference_table_key]
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

    def _update_statistics(self):
        """
        This function computes statistics
        with results currently in the result
        manager's heap
        """
        def _update_stats(statistics, result_heap, stats_key):
            passing_measurements = 0
            failing_measurements = 0
            total_configs = 0
            for result in result_heap.results():
                total_configs += 1
                passing_measurements += len(result.passing_measurements())
                failing_measurements += len(result.failing_measurements())

            statistics.set_total_configurations(stats_key, total_configs)
            statistics.set_passing_measurements(stats_key,
                                                passing_measurements)
            statistics.set_failing_measurements(stats_key,
                                                failing_measurements)

        for model_name, result_heap in self._per_model_sorted_results.items():
            _update_stats(self._statistics, result_heap, model_name)

        _update_stats(self._statistics, self._across_model_sorted_results,
                      TOP_MODELS_REPORT_KEY)
