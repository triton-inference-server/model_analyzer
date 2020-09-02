/*****************************************************************************
Copyright 2020, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*****************************************************************************/

using CommandLine;
using ModelAnalyzer.Client;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;

namespace ModelAnalyzer
{
    /// <summary>
    /// Class for launching Model Analyzer
    /// </summary>
    public class Program
    {
        /// <summary>
        /// Command line parser for Model Analyzer
        /// </summary>
        class Options
        {
            [Option('b', "batch", Separator = ',', Required = false, Default = new[] { 1 },
                HelpText = "Specifies comma-delimited list of batch sizes (default: 1)")]
            public IList<int> BatchSizes { get; set; }

            [Option('c', "concurrency", Separator = ',', Required = false, Default = new[] { 1 },
                HelpText = "Specifies comma-delimited list of concurrency values (default: 1)")]
            public IList<int> ConcurrencyValues { get; set; }

            [Option("export", Required = false, HelpText = "Enables exporting metrics to a CSV file")]
            public bool ExportFlag { get; set; }

            [Option('e', "export-path", Required = false, Default = "", HelpText = "Specifies the filepath for the export metrics")]
            public string ExportPath { get; set; }

            [Option("filename-model", Required = false, Default = "metrics-model.csv", HelpText = "Specifies filename for model running metrics")]
            public string FilenameModel { get; set; }

            [Option("filename-server-only", Required = false, Default = "metrics-server-only.csv", HelpText = "Specifies filename for server-only metrics")]
            public string FilenameServerOnly { get; set; }

            [Option('r', "max-retry", Required = false, HelpText = "Specifies the max seconds for any retry attempt")]
            public int MaxRetrySec { get; set; }

            [Option('m', "model-names", Separator = ',', Required = true, HelpText = "Specifies comma-delimited list of model names")]
            public IList<string> ModelNames { get; set; }

            [Option('p', "perf-buffer", Required = false, HelpText = "Specifies the number of seconds perf client runs model before gathering metrics")]
            public int PerfClientBufferSec { get; set; }

            [Option('d', "base-duration", Required = false, HelpText = "Specifies how long to gather server-only metrics")]
            public int ServerMetricsDuration { get; set; }

            [Option('v', "triton-version", Required = false, HelpText = "Specifies Triton version")]
            public string TritonVersion { get; set; }

            [Option("frequency-ms", Required = false, HelpText = "Specifies frequency of metric gathering in milliseconds")]
            public int UpdateFrequencyMs { get; set; }

        }

        [Verb("cli", isDefault: true, HelpText = "Runs Model Analyzer as a command line interface")]
        class CLIOptions : Options
        {

            [Option('f', "model-folder", Required = true, HelpText = "Specifies the absolute filepath of the models")]
            public string ModelFolder { get; set; }

        }

        [Verb("kubernetes", HelpText = "Runs Model Analyzer in a Kubernetes pod")]
        class K8sOptions : Options
        {

        }


        /// <summary>
        /// Parses command line argument options for the CLI/Docker program.
        /// </summary>
        /// <param name="options">Parsed options.</param>
        /// <returns>Model Analyzer configuration with parsed options.</returns>
        private ModelAnalyzerConfig ParseCLI(CLIOptions options)
        {
            var analyzerConfig = ParseBaseOptions(options);

            if (ModelAnalyzer.IsServerRunning())
                throw new ArgumentException("Inference server is already running on the default port");

            if (!Directory.Exists(options.ModelFolder))
                throw new ArgumentException("Model folder does not exist");

            analyzerConfig.ModelFolder = options.ModelFolder;

            return analyzerConfig;
        }

        /// <summary>
        /// Parses command line argument options for the Kubernetes program.
        /// </summary>
        /// <param name="options">Parsed options.</param>
        /// <returns>Model Analyzer configuration with parsed options.</returns>
        private ModelAnalyzerConfig ParseK8s(K8sOptions options)
        {
            return ParseBaseOptions(options);
        }


        /// <summary>
        /// Parses command line argument options
        /// </summary>
        /// <param name="options">Options to be parsed.</param>
        /// <returns>Model Analyzer configuration with parsed options.</returns>
        private ModelAnalyzerConfig ParseBaseOptions(Options options)
        {
            var analyzerConfig = new ModelAnalyzerConfig();



            if (!options.ExportFlag)
            {
                if (!string.IsNullOrWhiteSpace(options.ExportPath))
                    Console.WriteLine("Export-path specified without --export flag: skipping exporting metrics");
            }
            else
            {
                if (!string.IsNullOrEmpty(options.ExportPath) && !Directory.Exists(options.ExportPath))
                    throw new ArgumentException("Export path does not exist");
            }

            if (options.BatchSizes.Count != 0)
                analyzerConfig.BatchSize = options.BatchSizes;

            if (options.ConcurrencyValues.Count != 0)
                analyzerConfig.ConcurrencyValues = options.ConcurrencyValues;

            if (options.MaxRetrySec > 0)
                analyzerConfig.MaxRetryTime = TimeSpan.FromSeconds(options.MaxRetrySec);

            if (options.PerfClientBufferSec > 0)
                analyzerConfig.PerfClientBufferTime = TimeSpan.FromSeconds(options.PerfClientBufferSec);

            if (options.ServerMetricsDuration < 0)
                options.ServerMetricsDuration = 5;

            if (!string.IsNullOrWhiteSpace(options.TritonVersion))
                analyzerConfig.TritonVersion = options.TritonVersion;

            if (options.UpdateFrequencyMs > 0)
                analyzerConfig.UpdateFrequency = TimeSpan.FromMilliseconds(options.UpdateFrequencyMs);

            return analyzerConfig;
        }

        /// <summary>
        /// Exports metrics from MetricsCollector to standard out and/or file
        /// </summary>
        /// <param name="options">Parsed options.</param>
        /// <param name="metricsCollectorServerOnly">Metrics collector for server only.</param>
        /// <param name="metricsCollectorModel">Metrics collector for model.</param>
        private static void ExportMetrics(Options options, MetricsCollector metricsCollectorServerOnly, MetricsCollector metricsCollectorModel)
        {
            //Write metrics to screen
            Console.WriteLine("\nServer Only:");
            metricsCollectorServerOnly.ExportMetrics();
            Console.WriteLine("\nModels:");
            metricsCollectorModel.ExportMetrics();

            //Write metrics to file
            if (options.ExportFlag)
            {
                metricsCollectorServerOnly.ExportMetrics(Path.Combine(options.ExportPath, options.FilenameServerOnly));
                metricsCollectorModel.ExportMetrics(Path.Combine(options.ExportPath, options.FilenameModel));
            }
        }

        private static int Main(string[] args)
        {
            //Disable default help text.
            var parser = new CommandLine.Parser(with => with.HelpWriter = null);

            var program = new Program();

            return Parser.Default.ParseArguments<CLIOptions, K8sOptions>(args)
                .MapResult(
                (CLIOptions options) =>
                {
                    var analyzerConfig = program.ParseCLI(options);

                    //Initialize metrics collectors
                    var collectorConfig = new MetricsCollectorConfig()
                    {
                        RunLength = TimeSpan.FromSeconds(options.ServerMetricsDuration),
                        UpdateFrequency = analyzerConfig.UpdateFrequency,
                    };

                    var metricsCollectorServerOnly = new MetricsCollector(collectorConfig);
                    var metricsCollectorModel = new MetricsCollector(collectorConfig);

                    Console.CancelKeyPress += delegate
                    {
                        ExportMetrics(options, metricsCollectorServerOnly, metricsCollectorModel);
                    };

                    try
                    {
                        foreach (var model in options.ModelNames)
                        {
                            try
                            {
                                while (ModelAnalyzer.IsServerRunning())
                                {
                                    Thread.Sleep(TimeSpan.FromSeconds(1));
                                }

                                analyzerConfig.ModelName = model;
                                var analyzer = new ModelAnalyzer(analyzerConfig, metricsCollectorServerOnly, metricsCollectorModel);
                                analyzer.RunLocal();
                            }
                            catch (ModelLoadException)
                            {
                                Console.WriteLine($"Failed to load {model} on inference server: skipping model");
                            }
                            catch (Exception exception)
                            {
                                Console.WriteLine(exception.ToString());
                            }
                        }
                    }
                    finally
                    {
                        ExportMetrics(options, metricsCollectorServerOnly, metricsCollectorModel);
                    }

                    return 0;
                },
                (K8sOptions options) =>
                {
                    var analyzerConfig = program.ParseK8s(options);

                    //Initialize metrics collectors
                    var collectorConfig = new MetricsCollectorConfig()
                    {
                        RunLength = TimeSpan.FromSeconds(options.ServerMetricsDuration),
                        UpdateFrequency = analyzerConfig.UpdateFrequency,
                    };

                    var metricsCollectorServerOnly = new MetricsCollector(collectorConfig);
                    var metricsCollectorModel = new MetricsCollector(collectorConfig);

                    Console.CancelKeyPress += delegate
                    {
                        ExportMetrics(options, metricsCollectorServerOnly, metricsCollectorModel);
                    };

                    try
                    {
                        foreach (var model in options.ModelNames)
                        {
                            try
                            {
                                analyzerConfig.ModelName = model;
                                var analyzer = new ModelAnalyzer(analyzerConfig, metricsCollectorServerOnly, metricsCollectorModel);
                                analyzer.RunK8s();
                            }
                            catch (ModelLoadException)
                            {
                                Console.WriteLine($"Failed to load {model} on inference server: skipping model");
                            }
                            catch (Exception exception)
                            {
                                Console.WriteLine(exception.ToString());
                            }
                        }
                    }
                    finally
                    {
                        ExportMetrics(options, metricsCollectorServerOnly, metricsCollectorModel);
                    }

                    return 0;
                },
                errs =>
                {
                    return -1;
                }
                );
        }
    }
}