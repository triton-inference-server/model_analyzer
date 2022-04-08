# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.result.result_statistics import ResultStatistics
from model_analyzer.output.file_writer import FileWriter
from model_analyzer.constants import LOGGER_NAME, TOP_MODELS_REPORT_KEY
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from .result_heap import ResultHeap
from .result_table import ResultTable
from .run_config_result_comparator import RunConfigResultComparator
from .run_config_result import RunConfigResult
from .results import Results

import re
import os
import heapq
from collections import defaultdict
import logging

logger = logging.getLogger(LOGGER_NAME)


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
        'satisfies_constraints': 'Satisfies Constraints',
        'gpu_uuid': 'GPU UUID'
    }

    server_only_table_key = 'server_gpu_metrics'
    model_gpu_table_key = 'model_gpu_metrics'
    model_inference_table_key = 'model_inference_metrics'
    backend_parameter_key_prefix = 'backend_parameter/'

    def __init__(self, config, state_manager):
        """
        Parameters
        ----------
        config :ConfigCommandProfile
            the model analyzer config
        state_manager: AnalyzerStateManager
            The object that allows control and update of state
        """

        self._config = config
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

    def _init_state(self):
        """
        Sets ResultManager object managed
        state variables in AnalyerState
        """

        self._state_manager.set_state_variable('ResultManager.results',
                                               Results())
        self._state_manager.set_state_variable('ResultManager.server_only_data',
                                               {})

    def _create_server_table(self):
        # Server only
        server_output_headers = []
        server_output_fields = []
        for server_output_field in self._config.server_output_fields:
            if server_output_field in self.headers:
                server_output_headers.append(self.headers[server_output_field])
            elif server_output_field in self._gpu_metrics_to_headers:
                server_output_headers.append(
                    self._gpu_metrics_to_headers[server_output_field])
            else:
                logger.warning(
                    f'Server output field "{server_output_field}", has no data')
                continue
            server_output_fields.append(server_output_field)

        self._add_result_table(table_key=self.server_only_table_key,
                               title='Server Only',
                               headers=server_output_headers)
        self._server_output_fields = server_output_fields

    def _create_inference_table(self):
        # Inference only
        inference_output_headers = []
        inference_output_fields = []
        for inference_output_field in self._config.inference_output_fields:
            if inference_output_field in self.headers:
                inference_output_headers.append(
                    self.headers[inference_output_field])
            elif inference_output_field in self._non_gpu_metrics_to_headers:
                inference_output_headers.append(
                    self._non_gpu_metrics_to_headers[inference_output_field])
            elif inference_output_field.startswith(
                    self.backend_parameter_key_prefix):
                inference_output_headers.append(inference_output_field)
            else:
                logger.warning(
                    f'Inference output field "{inference_output_field}", has no data'
                )
                continue
            inference_output_fields.append(inference_output_field)

        self._inference_output_fields = inference_output_fields
        self._add_result_table(
            table_key=self.model_inference_table_key,
            title='Models (Inference)',
            headers=inference_output_headers,
        )

    def _create_gpu_table(self):
        gpu_output_headers = []
        gpu_output_fields = []
        for gpu_output_field in self._config.gpu_output_fields:
            if gpu_output_field in self.headers:
                gpu_output_headers.append(self.headers[gpu_output_field])
            elif gpu_output_field in self._gpu_metrics_to_headers:
                gpu_output_headers.append(
                    self._gpu_metrics_to_headers[gpu_output_field])
            else:
                logger.warning(
                    f'GPU output field "{gpu_output_field}", has no data')
                continue
            gpu_output_fields.append(gpu_output_field)

        self._gpu_output_fields = gpu_output_fields
        # Model GPU Metrics
        self._add_result_table(table_key=self.model_gpu_table_key,
                               title='Models (GPU Metrics)',
                               headers=gpu_output_headers)

    def create_tables(self,
                      gpu_specific_metrics=None,
                      non_gpu_specific_metrics=None):
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

        # Finds which metric(s) are actually collected during profile phase.
        # Since a profile phase can be run twice with different metric(s)
        # being collected.
        gpu_specific_metrics_from_measurements = {}
        non_gpu_specific_metrics_from_measurements = {}
        # Find metrics if one or more of them is not provided
        if gpu_specific_metrics == None or non_gpu_specific_metrics == None:
            # Server data
            data = self._state_manager.get_state_variable(
                "ResultManager.server_only_data")
            for gpu_uuid, gpu_metrics in data.items():
                for gpu_metric in gpu_metrics:
                    if gpu_metric.tag not in gpu_specific_metrics_from_measurements:
                        gpu_specific_metrics_from_measurements[
                            gpu_metric.tag] = gpu_metric
            # Measurements
            results = self._state_manager.get_state_variable(
                "ResultManager.results")

            for run_config_measurement in results.get_list_of_run_config_measurements(
            ):
                for gpu_uuid, gpu_metrics in run_config_measurement.gpu_data(
                ).items():
                    for gpu_metric in gpu_metrics:
                        if gpu_metric.tag not in gpu_specific_metrics_from_measurements:
                            gpu_specific_metrics_from_measurements[
                                gpu_metric.tag] = gpu_metric

                for non_gpu_metric_list in run_config_measurement.non_gpu_data(
                ):
                    for non_gpu_metric in non_gpu_metric_list:
                        if non_gpu_metric.tag not in non_gpu_specific_metrics_from_measurements:
                            non_gpu_specific_metrics_from_measurements[
                                non_gpu_metric.tag] = non_gpu_metric

        # Update not provided metric(s)
        if gpu_specific_metrics == None:
            gpu_specific_metrics = []
            for metric_tag, metric in gpu_specific_metrics_from_measurements.items(
            ):
                gpu_specific_metrics.append(metric)
        if non_gpu_specific_metrics == None:
            non_gpu_specific_metrics = []
            for metric_tag, metric in non_gpu_specific_metrics_from_measurements.items(
            ):
                non_gpu_specific_metrics.append(metric)

        # Add metric tag to header mapping
        for metric in gpu_specific_metrics:
            self._gpu_metrics_to_headers[metric.tag] = metric.header()
        for metric in non_gpu_specific_metrics:
            self._non_gpu_metrics_to_headers[metric.tag] = metric.header()

        self._create_inference_table()
        self._create_gpu_table()
        self._create_server_table()

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

        self._state_manager.set_state_variable('ResultManager.server_only_data',
                                               data)

    def add_run_config_measurement(self, run_config, run_config_measurement):
        """
        This function adds model inference
        measurements to the required result

        Parameters
        ----------
        run_config : RunConfig
            Contains the parameters used to generate the measurment
        run_config_measurement: RunConfigMeasurement
            the measurement to be added
        """

        # Get reference to results state and modify it
        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        results.add_run_config_measurement(run_config, run_config_measurement)

        # Use set_state_variable to record that state may have been changed
        self._state_manager.set_state_variable(name='ResultManager.results',
                                               value=results)

    def compile_and_sort_results(self):
        """
        Collects objectives and constraints for
        each model, constructs results from the
        measurements obtained, and sorts and 
        filters them according to constraints
        and objectives.
        """

        # Collect objectives and constraints
        comparators = {}
        constraints = {}
        for model in self._config.analysis_models:
            # TODO-TMA-570: Making a list for now, but objectives() should return a list of objectives
            # (one per model)
            comparators[model.model_name()] = RunConfigResultComparator(
                metric_objectives_list=[model.objectives()])
            constraints[model.model_name()] = model.constraints()

        # Construct and add results to individual result heaps as well as global result heap
        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        analysis_model_names = [
            model.model_name() for model in self._config.analysis_models
        ]
        for model_name in analysis_model_names:
            model_measurements = results.get_model_measurements_dict(model_name)

            # TODO-TMA-570: Model config should be a list
            for (run_config,
                 run_config_measurements) in model_measurements.values():
                run_config_result = RunConfigResult(
                    model_name=model_name,
                    run_config=run_config,
                    comparator=comparators[model_name],
                    constraints=constraints[model_name])

                for run_config_measurement in run_config_measurements.values():
                    run_config_measurement.set_metric_weightings(
                        comparators[model_name]._metric_weights)

                    run_config_result.add_run_config_measurement(
                        run_config_measurement)

                self._per_model_sorted_results[model_name].add_result(
                    run_config_result)
                self._across_model_sorted_results.add_result(run_config_result)

    def get_model_configs_run_config_measurements(self, model_variants_name):
        """
        Unsorted list of RunConfigMeasurements for a config

        Parameters
        ----------
        model_variants_name: str

        Returns
        -------
        (RunConfig, list of RunConfigMeasurements)
            The measurements for a particular config, in the order
            they were obtained.
        """

        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        # Name format is <base_model_name>_config_<number_or_default>
        #
        model_name = model_variants_name.rsplit('_', 2)[0]

        # Remote mode has model_name == model_config_name
        #
        if not results.contains_model(model_name):
            model_name = model_variants_name

        if results.contains_model(
                model_name) and results.contains_model_variant(
                    model_name, model_variants_name):
            return results.get_all_model_variant_measurements(
                model_name, model_variants_name)
        else:
            raise TritonModelAnalyzerException(
                f"RunConfig {model_variants_name} requested for report step but no results were found. "
                "Double check the name and ensure that this model config was actually profiled."
            )

    def top_n_results(self, model_name=None, n=-1, include_default=False):
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
        include_default : bool
            If true, the model's default config results will
            be included in the returned results. In the case
            that the default isn't one of the top n results,
            then n+1 results will be returned
        Returns
        -------
        list of RunConfigResults
            The n best results for this model,
            must all be passing results
        """

        if model_name:
            result_heap = self._per_model_sorted_results[model_name]
        else:
            result_heap = self._across_model_sorted_results
        results = result_heap.top_n_results(n)

        if include_default:
            self._add_default_to_results(model_name, results, result_heap)

        return results

    def tabulate_results(self):
        """
        The function called at the end of all runs
        FOR ALL MODELs that compiles all results and
        dumps the data into tables for exporting.
        """

        self._add_server_data()

        # Fill rows in descending order
        for result_heap in self._per_model_sorted_results.values():
            while not result_heap.empty():
                self._tabulate_measurements(result_heap.next_best_result())

    def _tabulate_measurements(self, run_config_result):
        """
        checks RunConfigMeasurements against constraints,
        and puts them into the correct (passing or failing)
        table
        """

        # TODO-TMA-570: This needs to be updated because there will be multiple model configs
        model_name = run_config_result.model_name()
        model_config = run_config_result.run_config().model_run_configs(
        )[0].model_config()

        instance_group = model_config.instance_group_string()
        dynamic_batching = model_config.dynamic_batching_string()
        cpu_only = run_config_result.run_config().cpu_only()
        backend_parameters = model_config._model_config.parameters

        passing_measurements = run_config_result.passing_measurements()
        failing_measurements = run_config_result.failing_measurements()

        for (measurements, passes) in [(passing_measurements, True),
                                       (failing_measurements, False)]:
            while measurements:
                next_best_measurement = heapq.heappop(measurements)
                self._tabulate_measurement(
                    model_name=model_name,
                    instance_group=instance_group,
                    dynamic_batching=dynamic_batching,
                    run_config_measurement=next_best_measurement,
                    passes=passes,
                    cpu_only=cpu_only,
                    backend_parameters=backend_parameters)

    def _tabulate_measurement(self, model_name, instance_group,
                              dynamic_batching, run_config_measurement, passes,
                              cpu_only, backend_parameters):
        """
        Add a single RunConfigMeasurement to the specified
        table
        """

        # TODO-TMA-570: This needs to be updated because there will be multiple model configs
        model_config_name = run_config_measurement._model_config_measurements[
            0].model_config_name()

        # TODO-TMA-570: Need to add accessor function to extract the PA parameters
        model_specific_pa_params = run_config_measurement._model_config_measurements[
            0].model_specific_pa_params()
        batch_size = model_specific_pa_params['batch-size']
        concurrency = model_specific_pa_params['concurrency-range']

        satisfies = "Yes" if passes else "No"

        # Non GPU specific data
        inference_fields = self._inference_output_fields
        inference_row = self._get_common_row_items(
            inference_fields, batch_size, concurrency, satisfies, model_name,
            model_config_name, dynamic_batching, instance_group,
            backend_parameters)

        # TODO-TMA-570: This needs to be examined for correctness
        for metric_list in run_config_measurement.non_gpu_data():
            for metric in metric_list:
                metric_tag_index = self._find_index_for_field(
                    inference_fields, metric.tag)

                if metric_tag_index is not None:
                    inference_row[metric_tag_index] = round(metric.value(), 1)

        self._result_tables[self.model_inference_table_key].insert_row_by_index(
            inference_row)

        # GPU specific data (only put measurement if not cpu only)
        if not cpu_only:
            for gpu_uuid, metrics in run_config_measurement.gpu_data().items():
                gpu_fields = self._gpu_output_fields
                gpu_row = self._get_common_row_items(
                    gpu_fields, batch_size, concurrency, satisfies, model_name,
                    model_config_name, dynamic_batching, instance_group)
                gpu_uuid_index = self._find_index_for_field(
                    gpu_fields, 'gpu_uuid')
                if gpu_uuid_index is not None:
                    gpu_row[gpu_uuid_index] = gpu_uuid
                for metric in metrics:
                    metric_tag_index = self._find_index_for_field(
                        gpu_fields, metric.tag)
                    if metric_tag_index is not None:
                        gpu_row[metric_tag_index] = round(metric.value(), 1)
                self._result_tables[
                    self.model_gpu_table_key].insert_row_by_index(row=gpu_row)

    def _get_common_row_items(self,
                              fields,
                              batch_size,
                              concurrency,
                              satisfies,
                              model_name,
                              model_config_path,
                              dynamic_batching,
                              instance_group,
                              backend_parameters=None):
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

        # Instance Group
        instance_group_idx = self._find_index_for_field(fields,
                                                        'instance_group')
        if instance_group_idx is not None:
            row[instance_group_idx] = instance_group

        # Backend parameters
        if backend_parameters is not None:
            for key in fields:
                if key.startswith(self.backend_parameter_key_prefix):
                    backend_parameter_key = key.replace(
                        self.backend_parameter_key_prefix, '')
                    backend_parameter_idx = self._find_index_for_field(
                        fields, key)

                    if backend_parameter_idx is not None and \
                        backend_parameter_key in backend_parameters:
                        row[backend_parameter_idx] = backend_parameters[
                            backend_parameter_key].string_value

        return row

    def _add_server_data(self):
        """
        Adds data to directly to the server only table

        Parameters
        ----------
        data : dict
            keys are gpu ids and values are lists of metric values
        """

        server_fields = self._server_output_fields
        server_only_data = self._state_manager.get_state_variable(
            'ResultManager.server_only_data')

        for gpu_uuid, metrics in server_only_data.items():
            data_row = [None] * len(server_fields)

            model_name_index = self._find_index_for_field(
                server_fields, 'model_name')
            if model_name_index is not None:
                data_row[model_name_index] = 'triton-server'

            gpu_uuid_index = self._find_index_for_field(server_fields,
                                                        'gpu_uuid')
            if gpu_uuid_index is not None:
                data_row[gpu_uuid_index] = gpu_uuid

            for metric in metrics:
                metric_tag_index = self._find_index_for_field(
                    server_fields, metric.tag)

                if metric_tag_index is not None:
                    data_row[metric_tag_index] = round(metric.value(), 1)
            self._result_tables[self.server_only_table_key].insert_row_by_index(
                data_row)

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

    def write_results(self):
        """
        Writes table to console
        """

        self._write_results(writer=FileWriter(), column_separator=' ')

    def export_results(self):
        """
        Makes calls to _write_results out to streams or files. If
        exporting results is requested, uses a FileWriter for specified output
        files.
        """

        # Results exported to export_path/results
        results_export_directory = os.path.join(self._config.export_path,
                                                'results')
        os.makedirs(results_export_directory, exist_ok=True)

        # Configure server only results path and export results
        server_metrics_path = os.path.join(results_export_directory,
                                           self._config.filename_server_only)
        logger.info(f"Exporting server only metrics to {server_metrics_path}")
        self._export_server_only_csv(
            writer=FileWriter(filename=server_metrics_path),
            column_separator=',')

        # Configure model metrics results path and export results
        metrics_inference_path = os.path.join(
            results_export_directory, self._config.filename_model_inference)
        metrics_gpu_path = os.path.join(results_export_directory,
                                        self._config.filename_model_gpu)
        logger.info(f"Exporting inference metrics to {metrics_inference_path}")
        logger.info(f"Exporting GPU metrics to {metrics_gpu_path}")
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
                                          ignore_widths=ignore_widths) + "\n\n")

    def _add_default_to_results(self, model_name, results, result_heap):
        '''
        If default config is already in results, keep it there. Else, find and
        add it from the result heap
        '''
        if not model_name:
            return

        # TODO-TMA-570: This logic needs to be changed for multi-model
        for run_config_result in results:
            if run_config_result.run_config().model_variants_name(
            ) == f"{model_name}_config_default":
                return

        # TODO-TMA-570: This logic needs to be changed for multi-model
        for run_config_result in result_heap.results():
            if run_config_result.run_config().model_variants_name(
            ) == f"{model_name}_config_default":
                results.append(run_config_result)
                return

    def get_result_statistics(self):
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
            statistics.set_passing_measurements(stats_key, passing_measurements)
            statistics.set_failing_measurements(stats_key, failing_measurements)

        result_stats = ResultStatistics()
        for model_name, result_heap in self._per_model_sorted_results.items():
            _update_stats(result_stats, result_heap, model_name)

        _update_stats(result_stats, self._across_model_sorted_results,
                      TOP_MODELS_REPORT_KEY)

        return result_stats
