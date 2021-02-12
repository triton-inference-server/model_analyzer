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

import heapq

from .result_table import ResultTable
from .run_result import RunResult
from .constraint_manager import ConstraintManager
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class ResultManager:
    """
    This class provides methods to create, and add to 
    ResultTables. Each ResultTable holds results from
    multiple runs.
    """

    non_gpu_specific_headers = ['Model', 'Batch', 'Concurrency']
    gpu_specific_headers = ['Model', 'GPU ID', 'Batch', 'Concurrency']
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
        self._current_run_results = {}
        self._result_comparator = None

        # Results are stored in a heap queue
        self._results = []

    def set_constraints_and_comparator(self, constraints, comparator):
        """
        Sets the ResultComparator for all the results
        this ResultManager will construct

        Parameters
        ----------
        constraints : dict
            The constraints that determine whether
            a measurement can be used.
        comparator : ResultComparator
            the result comparator function object that can
            compare two results.
        """

        self._constraint_manager = ConstraintManager(constraints=constraints)
        self._result_comparator = comparator

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
                               headers=self.gpu_specific_headers,
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
            run
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

    def add_model_data(self, measurement):
        """
        This function adds model inference
        measurements to the result, not directly
        to a table.

        Parameters
        ----------
        measurement : Measurement
            The measurements from the metrics manager,
            actual values from the monitors
        """

        self._current_run_result.add_data(measurement=measurement)

    def complete_result(self):
        """
        Submit the current RunResults into
        the ResultTable
        """

        heapq.heappush(self._results, self._current_run_result)

    def compile_results(self):
        """
        The function called at the end of all runs 
        FOR A MODEL that compiles all result and 
        dumps the data into tables for exporting.
        """

        # Fill rows in descending order
        while self._results:
            next_best_result = heapq.heappop(self._results)
            measurements = next_best_result.get_measurements()
            self._compile_measurements(measurements)

    def _compile_measurements(self, measurements):
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
                    measurement=next_best_measurement,
                    gpu_table_key=self.model_gpu_table_passing_key,
                    inference_table_key=self.model_inference_table_passing_key)
            else:
                self._compile_measurement(
                    measurement=next_best_measurement,
                    gpu_table_key=self.model_gpu_table_failing_key,
                    inference_table_key=self.model_inference_table_failing_key)

    def _compile_measurement(self, measurement, gpu_table_key,
                             inference_table_key):
        """
        Add a single measurement to the specified
        table
        """

        perf_config = measurement.perf_config()
        model_name = perf_config['model-name']
        batch_size = perf_config['batch-size']
        concurrency = perf_config['concurrency-range']

        # Non GPU specific data
        inference_metrics = [model_name, batch_size, concurrency]
        inference_metrics += [
            metric.value() for metric in measurement.non_gpu_data()
        ]
        self._result_tables[inference_table_key].insert_row_by_index(
            row=inference_metrics)

        # GPU specific data
        for gpu_id, metrics in measurement.gpu_data().items():
            gpu_metrics = [model_name, gpu_id, batch_size, concurrency]
            gpu_metrics += [metric.value() for metric in metrics]
            self._result_tables[gpu_table_key].insert_row_by_index(
                row=gpu_metrics)

    def get_all_tables(self):
        """
        Returns
        -------
        dict 
            table keys and ResultTables
        """

        return self._result_tables

    def get_server_table(self):
        """
        Returns
        -------
        ResultTable
            The table corresponding to server only
            data
        """

        return self._get_table(self.server_only_table_key)

    def get_passing_model_tables(self):
        """
        Returns
        -------
        (ResultTable, ResultTable)
            The tables corresponding to the model inference
            data
        """

        return self._get_table(
            self.model_gpu_table_passing_key), self._get_table(
                self.model_inference_table_passing_key)

    def get_failing_model_tables(self):
        """
        Returns
        -------
        (ResultTable, ResultTable)
            The table corresponding to the model inference
            data
        """

        return self._get_table(
            self.model_gpu_table_failing_key), self._get_table(
                self.model_inference_table_failing_key)

    def _get_table(self, key):
        """
        Get a ResultTable by table key
        """

        if key not in self._result_tables:
            raise TritonModelAnalyzerException(
                f"Table with key '{key}' not found in ResultManager")
        return self._result_tables[key]

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
