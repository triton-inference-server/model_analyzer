# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time

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

        self.perf_analyzer_path = args.perf_analyzer_path
        self.base_duration = args.base_duration
        self.monitoring_interval = args.monitoring_interval
        self.monitoring_metrics = monitoring_metrics
        self.param_headers = ['Model', 'Batch', 'Concurrency']

        # Separates metric tags into perf_analyzer related and DCGM related tags
        self.dcgm_tags = []
        self.perf_tags = []
        for metric in self.monitoring_metrics:
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
        Runs the DCGM monitor on the triton
        server without the perf_analyzer

        Parameters
        ----------
        default_value : str
            The value to fill in for columns in the table
            that don't apply to profiling server only

        Raises
        ------
        TritonModelAnalyzerException
        """

        dcgm_monitor = DCGMMonitor(self.monitoring_interval, self.dcgm_tags)
        server_only_metrics = self._profile(perf_analyzer=None,
                                            dcgm_monitor=dcgm_monitor)

        # Model name here is triton-server, batch and concurrency are defaults
        output_row = ['triton-server', default_value, default_value]

        # add the obtained metrics
        for metric in self.monitoring_metrics:
            if metric in server_only_metrics:
                output_row.append(server_only_metrics[metric])
            else:
                output_row.append(default_value)
        self.tables["Server Only:"].add_row(output_row)
        dcgm_monitor.destroy()

    def profile_model(self, run_config, perf_output_writer=None):
        """
        Runs DCGMMonitor while running perf_analyzer
        with a specific set of arguments. This will
        profile model inferencing.

        Parameters
        ----------
        run_config : dict
            The keys are arguments to perf_analyzer
            The values are their values
        perf_output_writer : OutputWriter
            Writer that writes the output from  
            perf_analyzer to the output stream/file.
            If None, the output is not written

        Raises
        ------
        TritonModelAnalyzerException
        """

        dcgm_monitor = DCGMMonitor(self.monitoring_interval, self.dcgm_tags)
        perf_analyzer = PerfAnalyzer(
            path=self.perf_analyzer_path,
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
            model_metrics[metric] for metric in self.monitoring_metrics
        ]

        self.tables["Models:"].add_row(output_row)

        dcgm_monitor.destroy()

    def write_results(self, writer, column_separator, ignore_widths=False):
        """
        Writes the tables using the writer with the given
        column specifications.

        Parameters
        ----------
        writer : OutputWriter
            Used to write the result tables to an output stream
        column_separator : str
            The string that will be inserted between each column
            of the table
        ignore_widths : boolean
            If true, columns of the table will not be of fixed widths,
            they will vary with the contents.

        Raises
        ------
        TritonModelAnalyzerException
        """

        for table_name in self.tables:
            self._write_result(table_name, writer, column_separator,
                               ignore_widths)

    def export_server_only_csv(self,
                               writer,
                               column_separator,
                               ignore_widths=False):
        """
        Writes the server-only table as
        a csv file using the given writer
        
        Parameters
        ----------
        writer : OutputWriter
            Used to write the result tables to an output stream
        column_separator : str
            The string that will be inserted between each column
            of the table
        ignore_widths : boolean
            If true, columns of the table will not be of fixed widths,
            they will vary with the contents.

        Raises
        ------
        TritonModelAnalyzerException
        """

        self._write_result("Server Only:",
                           writer,
                           column_separator,
                           ignore_widths,
                           write_table_name=False)

    def export_model_csv(self, writer, column_separator, ignore_widths=False):
        """
        Writes the model table as
        a csv file using the given writer
        
        Parameters
        ----------
        writer : OutputWriter
            Used to write the result tables to an output stream
        column_separator : str
            The string that will be inserted between each column
            of the table
        ignore_widths : boolean
            If true, columns of the table will not be of fixed widths,
            they will vary with the contents.

        Raises
        ------
        TritonModelAnalyzerException
        """

        self._write_result("Models:",
                           writer,
                           column_separator,
                           ignore_widths,
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

        if ignore_widths:
            for header in self.tables[table_name].headers():
                self.tables[table_name].set_column_width_by_header(
                    header, None)
        if write_table_name:
            writer.write('\n'.join([
                table_name, self.tables[table_name].to_formatted_string(
                    separator=column_separator), "\n"
            ]))
        else:
            writer.write(self.tables[table_name].to_formatted_string(
                separator=column_separator) + "\n\n")

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
            time.sleep(self.base_duration)
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
        table_headers = self.param_headers[:]
        for metric in self.monitoring_metrics:
            if metric in self.dcgm_tags:
                table_headers.append(aggregation_tag + ' ' + metric.header())
            else:
                table_headers.append(metric.header())

        # Set column widths (Model colummn width is 28, rest are 2 more than length of header)
        column_widths = [28]
        column_widths += [len(header) + 2 for header in table_headers[1:]]
        return OutputTable(headers=table_headers, column_widths=column_widths)

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
