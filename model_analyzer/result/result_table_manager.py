# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from .result_utils import format_for_csv
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.output.file_writer import FileWriter
import os
import logging

logger = logging.getLogger(LOGGER_NAME)


class ResultTableManager:
    """
    This class provides methods to create, and add to
    ResultTables. Each ResultTable holds results from
    multiple runs.    
    """

    headers = {
        'model_name': 'Model',
        'batch_size': 'Batch',
        'concurrency': 'Concurrency',
        'request_rate': 'Request Rate',
        'model_config_path': 'Model Config Path',
        'instance_group': 'Instance Group',
        'max_batch_size': 'Max Batch Size',
        'satisfies_constraints': 'Satisfies Constraints',
        'gpu_uuid': 'GPU UUID'
    }

    server_only_table_key = 'server_gpu_metrics'
    model_gpu_table_key = 'model_gpu_metrics'
    model_inference_table_key = 'model_inference_metrics'
    backend_parameter_key_prefix = 'backend_parameter/'

    def __init__(self, config, result_manager):
        self._config = config
        self._result_manager = result_manager

        # Headers Dictionary and result tables
        self._gpu_metrics_to_headers = {}
        self._non_gpu_metrics_to_headers = {}
        self._result_tables = {}

    def create_tables(self):
        """
        Creates the inference, gpu, and server tables
        """
        self._determine_table_headers()

        self._create_inference_table()
        self._create_gpu_table()
        self._create_server_table()

    def tabulate_results(self):
        """
        The function called at the end of all runs
        FOR ALL MODELs that compiles all results and
        dumps the data into tables for exporting.
        """
        self._add_server_data()

        # Fill rows in descending order
        for model in self._result_manager.get_model_names():
            for result in self._result_manager.get_model_sorted_results(
                    model).results():
                self._tabulate_measurements(result)

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

        results_export_directory = os.path.join(self._config.export_path,
                                                'results')
        os.makedirs(results_export_directory, exist_ok=True)

        self._export_results(name="server only",
                             dir=results_export_directory,
                             filename=self._config.filename_server_only,
                             key=self.server_only_table_key)

        self._export_results(name="inference",
                             dir=results_export_directory,
                             filename=self._config.filename_model_inference,
                             key=self.model_inference_table_key)

        self._export_results(name="GPU",
                             dir=results_export_directory,
                             filename=self._config.filename_model_gpu,
                             key=self.model_gpu_table_key)

    def _export_results(self, name, dir, filename, key):
        table = self._result_tables[key]
        if table.size():
            outfile = os.path.join(dir, filename)
            logger.info(f"Exporting {name} metrics to {outfile}")
            self._write_result(table=table,
                               writer=FileWriter(filename=outfile),
                               column_separator=',',
                               ignore_widths=True,
                               include_title=False)

    def _determine_table_headers(self):
        # Finds which metric(s) are actually collected during profile phase.
        # Since a profile phase can be run twice with different metric(s)
        # being collected.
        gpu_metrics_from_measurements = {}
        non_gpu_metrics_from_measurements = {}

        # Server data
        data = self._result_manager.get_server_only_data()
        for gpu_metrics in data.values():
            for gpu_metric in gpu_metrics:
                if gpu_metric.tag not in gpu_metrics_from_measurements:
                    gpu_metrics_from_measurements[gpu_metric.tag] = gpu_metric

        # Measurements
        results = self._result_manager.get_results()

        for run_config_measurement in results.get_list_of_run_config_measurements(
        ):
            for gpu_metrics in run_config_measurement.gpu_data().values():
                for gpu_metric in gpu_metrics:
                    if gpu_metric.tag not in gpu_metrics_from_measurements:
                        gpu_metrics_from_measurements[
                            gpu_metric.tag] = gpu_metric

            for non_gpu_metric_list in run_config_measurement.non_gpu_data():
                for non_gpu_metric in non_gpu_metric_list:
                    if non_gpu_metric.tag not in non_gpu_metrics_from_measurements:
                        non_gpu_metrics_from_measurements[
                            non_gpu_metric.tag] = non_gpu_metric

        gpu_specific_metrics = gpu_metrics_from_measurements.values()
        non_gpu_specific_metrics = non_gpu_metrics_from_measurements.values()

        # Add metric tags to header mappings
        for metric in gpu_specific_metrics:
            self._gpu_metrics_to_headers[metric.tag] = metric.header()
        for metric in non_gpu_specific_metrics:
            self._non_gpu_metrics_to_headers[metric.tag] = metric.header()

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

    def _find_index_for_field(self, fields, field_name):
        try:
            index = fields.index(field_name)
            return index
        except ValueError:
            return None

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

    def _get_gpu_count(self):
        return self._result_tables[self.server_only_table_key].size()

    def _add_server_data(self):
        """
        Adds data to directly to the server only table

        Parameters
        ----------
        data : dict
            keys are gpu ids and values are lists of metric values
        """

        server_fields = self._server_output_fields
        server_only_data = self._result_manager.get_server_only_data()

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

        self._result_tables[table_key] = ResultTable(headers=headers,
                                                     title=title)

    def _tabulate_measurements(self, run_config_result):
        """
        checks RunConfigMeasurements against constraints,
        and puts them into the correct (passing or failing)
        table
        """

        model_name = run_config_result.model_name()
        instance_groups, max_batch_sizes, dynamic_batchings, cpu_onlys, backend_parameters, composing_config_names = self._tabulate_measurements_setup(
            run_config_result)

        passing_measurements = run_config_result.passing_measurements()
        failing_measurements = run_config_result.failing_measurements()

        for (run_config_measurements, passes) in [(passing_measurements, True),
                                                  (failing_measurements, False)
                                                 ]:
            for run_config_measurement in run_config_measurements:
                self._tabulate_measurement(
                    model_name=model_name,
                    instance_groups=instance_groups,
                    max_batch_sizes=max_batch_sizes,
                    dynamic_batchings=dynamic_batchings,
                    run_config_measurement=run_config_measurement,
                    passes=passes,
                    cpu_onlys=cpu_onlys,
                    backend_parameters=backend_parameters,
                    composing_config_names=composing_config_names)

    def _tabulate_measurements_setup(self, run_config_result):
        if run_config_result.run_config().is_ensemble_model():
            model_configs = run_config_result.run_config().composing_configs()
            composing_config_names = [
                model_config.get_field("name") for model_config in model_configs
            ]
        else:
            model_configs = [
                model_run_configs.model_config() for model_run_configs in
                run_config_result.run_config().model_run_configs()
            ]

            composing_config_names = []

        instance_groups = [
            model_config.instance_group_string(self._get_gpu_count())
            for model_config in model_configs
        ]
        max_batch_sizes = [
            model_config.max_batch_size() for model_config in model_configs
        ]
        dynamic_batchings = [
            model_config.dynamic_batching_string()
            for model_config in model_configs
        ]
        cpu_onlys = [
            run_config_result.run_config().cpu_only()
            for model_config in model_configs
        ]
        backend_parameters = [
            model_config._model_config.parameters
            for model_config in model_configs
        ]

        return instance_groups, max_batch_sizes, dynamic_batchings, cpu_onlys, backend_parameters, composing_config_names

    def _tabulate_measurement(self, model_name, instance_groups,
                              max_batch_sizes, dynamic_batchings,
                              run_config_measurement, passes, cpu_onlys,
                              backend_parameters, composing_config_names):
        """
        Add a single RunConfigMeasurement to the specified
        table
        """

        model_config_name = run_config_measurement.model_variants_name()
        if composing_config_names:
            model_config_name = model_config_name + ": "
            for composing_config_name in composing_config_names:
                model_config_name = model_config_name + composing_config_name

                if composing_config_name != composing_config_names[-1]:
                    model_config_name = model_config_name + ", "

        model_specific_pa_params, batch_sizes, concurrencies, request_rates = self._tabulate_measurement_setup(
            run_config_measurement)

        satisfies = "Yes" if passes else "No"

        # Non GPU specific data
        inference_fields = self._inference_output_fields
        inference_row = self._get_common_row_items(
            inference_fields, batch_sizes, concurrencies, request_rates,
            satisfies, model_name, model_config_name, dynamic_batchings,
            instance_groups, max_batch_sizes, backend_parameters)

        self._populate_inference_rows(run_config_measurement, inference_fields,
                                      inference_row)

        self._result_tables[self.model_inference_table_key].insert_row_by_index(
            inference_row)

        # GPU specific data (only put measurement if not cpu only)
        if not any(cpu_onlys):
            for gpu_uuid, metrics in run_config_measurement.gpu_data().items():
                gpu_fields = self._gpu_output_fields

                gpu_row = self._get_common_row_items(
                    gpu_fields, batch_sizes, concurrencies, request_rates,
                    satisfies, model_name, model_config_name, dynamic_batchings,
                    instance_groups, max_batch_sizes)

                self._add_uuid_to_gpu_row(gpu_row, gpu_uuid, gpu_fields)
                self._add_metrics_to_gpu_row(gpu_row, metrics, gpu_fields)

                self._result_tables[
                    self.model_gpu_table_key].insert_row_by_index(row=gpu_row)

    def _tabulate_measurement_setup(self, run_config_measurement):
        model_specific_pa_params = run_config_measurement.model_specific_pa_params(
        )
        batch_sizes = [
            pa_params['batch-size']
            for pa_params in model_specific_pa_params
            if 'batch-size' in pa_params
        ]
        concurrencies = [
            pa_params['concurrency-range']
            for pa_params in model_specific_pa_params
            if 'concurrency-range' in pa_params
        ]
        request_rates = [
            pa_params['request-rate-range']
            for pa_params in model_specific_pa_params
            if 'request-rate-range' in pa_params
        ]

        return model_specific_pa_params, batch_sizes, concurrencies, request_rates

    def _populate_inference_rows(self, run_config_measurement, inference_fields,
                                 inference_row):
        # FIXME: TMA-686 - Need to figure out what to do if models have different tags
        for metric in run_config_measurement.non_gpu_data()[0]:
            metric_tag_index = self._find_index_for_field(
                inference_fields, metric.tag)
            if metric_tag_index is not None:
                inference_row[
                    metric_tag_index] = self._create_non_gpu_metric_row_entry(
                        run_config_measurement, metric)

    def _add_uuid_to_gpu_row(self, gpu_row, gpu_uuid, gpu_fields):
        gpu_uuid_index = self._find_index_for_field(gpu_fields, 'gpu_uuid')

        if gpu_uuid_index is not None:
            gpu_row[gpu_uuid_index] = gpu_uuid

    def _add_metrics_to_gpu_row(self, gpu_row, metrics, gpu_fields):
        for metric in metrics:
            metric_tag_index = self._find_index_for_field(
                gpu_fields, metric.tag)

            if metric_tag_index is not None:
                gpu_row[metric_tag_index] = round(metric.value(), 1)

    def _create_non_gpu_metric_row_entry(self, run_config_measurement, metric):
        metric_value = run_config_measurement.get_non_gpu_metric_value(
            metric.tag)
        non_gpu_metrics = run_config_measurement.get_non_gpu_metric(metric.tag)

        if len(non_gpu_metrics) > 1:
            rounded_non_gpu_metrics = [
                round(metric.value(), 1) for metric in
                run_config_measurement.get_non_gpu_metric(metric.tag)
            ]

            return format_for_csv(
                [round(metric_value, 1), rounded_non_gpu_metrics])

        else:
            return format_for_csv(round(metric_value, 1))

    def _get_common_row_items(self,
                              fields,
                              batch_sizes,
                              concurrencies,
                              request_rates,
                              satisfies,
                              model_name,
                              model_config_path,
                              dynamic_batchings,
                              instance_groups,
                              max_batch_sizes,
                              backend_parameters=None):
        row = [None] * len(fields)

        # Model Name
        model_name_index = self._find_index_for_field(fields, 'model_name')
        if model_name_index is not None:
            row[model_name_index] = format_for_csv(model_name)

        # Batch Size
        batch_size_index = self._find_index_for_field(fields, 'batch_size')
        if batch_size_index is not None:
            row[batch_size_index] = format_for_csv(batch_sizes)

        # Concurrency
        concurrency_index = self._find_index_for_field(fields, 'concurrency')
        if concurrency_index is not None:
            row[concurrency_index] = format_for_csv(concurrencies)

        # Request rate
        request_rate_index = self._find_index_for_field(fields, 'request_rate')
        if request_rate_index is not None:
            row[request_rate_index] = format_for_csv(request_rates)

        # Satisfies
        satisfies_constraints_index = self._find_index_for_field(
            fields, 'satisfies_constraints')
        if satisfies_constraints_index is not None:
            row[satisfies_constraints_index] = format_for_csv(satisfies)

        # Model Config Path
        model_config_path_idx = self._find_index_for_field(
            fields, 'model_config_path')
        if model_config_path_idx is not None:
            row[model_config_path_idx] = format_for_csv(model_config_path)

        # Instance Group
        instance_group_idx = self._find_index_for_field(fields,
                                                        'instance_group')
        if instance_group_idx is not None:
            row[instance_group_idx] = format_for_csv(instance_groups)

        # Max Batch Size
        max_batch_size_idx = self._find_index_for_field(fields,
                                                        'max_batch_size')
        if max_batch_size_idx is not None:
            row[max_batch_size_idx] = format_for_csv(max_batch_sizes)

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
