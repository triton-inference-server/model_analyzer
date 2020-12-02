# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import time
import logging

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from .monitor.dcgm.dcgm_monitor import DCGMMonitor
from .perf_analyzer.perf_analyzer import PerfAnalyzer
from .perf_analyzer.perf_config import PerfAnalyzerConfig
from .record.record_aggregator import RecordAggregator
from .record.gpu_free_memory import GPUFreeMemory
from .record.gpu_used_memory import GPUUsedMemory
from .record.perf_throughput import PerfThroughput
from .record.perf_latency import PerfLatency
from .output.output_table import OutputTable

logger = logging.getLogger(__name__)


class Analyzer:
    """
    A class responsible for coordinating
    the various components of the model_analyzer.
    Configured with metrics to monitor,
    exposes profiling and result writing
    methods.
    """

    def __init__(self, args, monitoring_metrics):
        """
        Parameters
        ----------
        args : namespace
            The arguments passed into the CLI
        monitoring_metrics : List of Record types
            The list of metric types to monitor.
        """

        self._perf_analyzer_path = args.perf_analyzer_path
        self._duration_seconds = args.duration_seconds
        self._monitoring_interval = args.monitoring_interval
        self._monitoring_metrics = monitoring_metrics
        self._param_headers = ['Model', 'Batch', 'Concurrency']
        self._gpus = args.gpus

        # Separates metric tags into perf_analyzer related and DCGM related tags
        self.dcgm_tags = []
        self.perf_tags = []
        for metric in self._monitoring_metrics:
            if metric in list(DCGMMonitor.model_analyzer_to_dcgm_field):
                self.dcgm_tags.append(metric)
            elif metric in PerfAnalyzer.perf_metrics:
                self.perf_tags.append(metric)

        self.tables = {
            "Server Only:": self._create_output_table(),
            "Models:": self._create_output_table()
        }

    def profile_server_only(self, default_value='0'):
        """
        Runs the DCGM monitor on the triton server without the perf_analyzer

        Parameters
        ----------
        default_value : str
            The value to fill in for columns in the table that don't apply to
            profiling server only

        Raises
        ------
        TritonModelAnalyzerException
        """

        logging.info('Profiling server only metrics...')
        dcgm_monitor = DCGMMonitor(self._gpus, self._monitoring_interval,
                                   self.dcgm_tags)
        server_only_metrics = self._profile(perf_analyzer=None,
                                            dcgm_monitor=dcgm_monitor)

        # Model name here is triton-server, batch and concurrency are defaults
        output_row = ['triton-server', default_value, default_value]

        # add the obtained metrics
        for metric in self._monitoring_metrics:
            if metric in server_only_metrics:
                output_row.append(server_only_metrics[metric])
            else:
                output_row.append(default_value)
        self.tables["Server Only:"].add_row(output_row)
        dcgm_monitor.destroy()

    def profile_model(self, run_config, perf_output_writer=None):
        """
        Runs DCGMMonitor while running perf_analyzer with a specific set of
        arguments. This will profile model inferencing.

        Parameters
        ----------
        run_config : dict
            The keys are arguments to perf_analyzer The values are their
            values
        perf_output_writer : OutputWriter
            Writer that writes the output from perf_analyzer to the output
            stream/file. If None, the output is not written

        Raises
        ------
        TritonModelAnalyzerException
        """

        logging.info(f"Profiling model {run_config['model-name']}...")
        dcgm_monitor = DCGMMonitor(self._gpus, self._monitoring_interval,
                                   self.dcgm_tags)
        perf_analyzer = PerfAnalyzer(
            path=self._perf_analyzer_path,
            config=self._create_perf_config(run_config))

        # Get metrics for model inference and write perf_output
        model_metrics = self._profile(perf_analyzer=perf_analyzer,
                                      dcgm_monitor=dcgm_monitor)
        if perf_output_writer:
            perf_output_writer.write(perf_analyzer.output() + '\n')

        output_row = [
            run_config['model-name'], run_config['batch-size'],
            run_config['concurrency-range']
        ]
        output_row += [
            model_metrics[metric] for metric in self._monitoring_metrics
        ]

        self.tables["Models:"].add_row(output_row)

        dcgm_monitor.destroy()

    def write_results(self, writer, column_separator):
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

        uniform_widths = self._get_max_width_across_tables()
        for table_name in self.tables:
            self._set_table_column_widths(table_name, uniform_widths)
            self._write_result(table_name,
                               writer,
                               column_separator,
                               ignore_widths=False)

    def export_server_only_csv(self, writer, column_separator):
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

        self._write_result("Server Only:",
                           writer,
                           column_separator,
                           ignore_widths=True,
                           write_table_name=False)

    def export_model_csv(self, writer, column_separator):
        """
        Writes the model table as a csv file using the given writer

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

        self._write_result("Models:",
                           writer,
                           column_separator,
                           ignore_widths=True,
                           write_table_name=False)

    def _write_result(self,
                      table_name,
                      writer,
                      column_separator,
                      ignore_widths=False,
                      write_table_name=True):
        """
        Utility function that writes any table
        """

        if write_table_name:
            writer.write('\n'.join([
                table_name, self.tables[table_name].to_formatted_string(
                    separator=column_separator, ignore_widths=ignore_widths),
                "\n"
            ]))
        else:
            writer.write(self.tables[table_name].to_formatted_string(
                separator=column_separator, ignore_widths=ignore_widths) +
                         "\n\n")

    def _profile(self, perf_analyzer, dcgm_monitor):
        """
        Utility function that runs the perf_analyzer
        and DCGMMonitor once.

        Raises
        ------
        TritonModelAnalyzerException
            if path to perf_analyzer binary could
            not be found.
        """

        # Start monitors and run perf_analyzer
        dcgm_monitor.start_recording_metrics()
        if perf_analyzer:
            try:
                perf_records = perf_analyzer.run(self.perf_tags)
            except FileNotFoundError as e:
                raise TritonModelAnalyzerException(
                    f"perf_analyzer binary not found : {e}")
        else:
            perf_records = []
            time.sleep(self._duration_seconds)
        dcgm_records = dcgm_monitor.stop_recording_metrics()

        # Insert all records into aggregator and get aggregated DCGM records
        record_aggregator = RecordAggregator()
        for record in perf_records + dcgm_records:
            record_aggregator.insert(record)
        return record_aggregator.aggregate()

    def _create_output_table(self, aggregation_tag='Max'):
        """
        Utility function that creates a table with column
        headers corresponding to perf_analyzer arguments
        and requested metrics.
        """

        # Create headers
        table_headers = self._param_headers[:]
        for metric in self._monitoring_metrics:
            if metric in self.dcgm_tags:
                table_headers.append(aggregation_tag + ' ' + metric.header())
            else:
                table_headers.append(metric.header())
        return OutputTable(headers=table_headers)

    def _create_perf_config(self, params):
        """
        Utility function for creating
        a PerfAnalyzerConfig from
        a dict of parameters.
        """

        config = PerfAnalyzerConfig()
        for param, value in params.items():
            config[param] = value
        return config

    def _get_max_width_across_tables(self):
        """
        Compares the width of all columns across all
        tables and returns a list of max widths
        """

        individual_widths = [
            self.tables[k].column_widths() for k in self.tables
        ]
        uniform_widths = individual_widths[0]
        for widths in individual_widths[1:]:
            uniform_widths = [
                max(uniform_widths[j], widths[j]) for j in range(len(widths))
            ]
        return uniform_widths

    def _set_table_column_widths(self, table_name, widths):
        """
        Sets the column widths of the table with given name
        by index correspoding to given list of widths.
        """

        for j in range(len(self.tables[table_name].column_widths())):
            self.tables[table_name].set_column_width_by_index(j, widths[j])
