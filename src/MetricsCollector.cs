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

using ModelAnalyzer.Metrics;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;

namespace ModelAnalyzer
{
    /// <summary>
    /// Class used to store configuration of Metrics Collector
    /// </summary>
    class MetricsCollectorConfig
    {
        /// <summary>
        /// Length to run Metrics Collector
        /// </summary>
        public TimeSpan RunLength { get; set; }

        /// <summary>
        /// Milliseconds to wait between each metric collection
        /// </summary>
        public TimeSpan UpdateFrequency { get; set; }
    }

    /// <summary>
    /// Class used to gather and store system metrics
    /// </summary>
    class MetricsCollector
    {
        /// <summary>
        /// Maximum number of attempts to clear the cached metric values
        /// </summary>
        const int MaxCacheClearAttempts = 30;

        /// <summary>
        /// Most recent sample stored in Metrics Collector
        /// </summary>
        private GpuMetricSample _LatestSample;

        /// <summary>
        /// Length to run Metrics Collector
        /// </summary>
        private readonly TimeSpan _RunLength;

        /// <summary>
        /// Milliseconds to wait between each metric collection
        /// </summary>
        private readonly TimeSpan _UpdateFrequency;

        /// <summary>
        /// _GpuMetrics object with latest gathered metrics
        /// </summary>
        readonly GpuMetrics _GpuMetrics = new GpuMetrics();

        /// <summary>
        /// List of metrics samples stored in Metrics Collector
        /// </summary>
        public readonly IList<GpuMetricSample> _MetricsSamples = new List<GpuMetricSample>();

        /// <summary>
        /// Object that collects metrics
        /// </summary>
        private readonly IMetricsWatcher _Watcher;

        /// <summary>
        /// Constructor for MetricsCollector
        /// </summary>
        public MetricsCollector(MetricsCollectorConfig config)
        {
            _RunLength = config.RunLength;
            _UpdateFrequency = config.UpdateFrequency;
            _Watcher = _GpuMetrics.GetWatcher("metrics", _UpdateFrequency);
        }

        /// <summary>
        /// Adds a new sample to Metrics Collector
        /// </summary>
        /// <param name="modelName">Name of model associated with sample.</param>
        /// <param name="batchSize">Number of models sent per batch to server.</param>
        /// <param name="concurrencyValue">Number of concurrent threads used by server.</param>
        /// <returns>Returns new sample.</returns>
        public GpuMetricSample AddNewSample(string modelName, int batchSize, int concurrencyValue)
        {
            _LatestSample = new GpuMetricSample(modelName, batchSize, concurrencyValue);
            _MetricsSamples.Add(_LatestSample);
            return _LatestSample;
        }

        /// <summary>
        /// Delete newest sample in Metrics Collector
        /// </summary>
        public void DeletesLatestSample()
        {
            _MetricsSamples.Remove(_LatestSample);
        }

        public void GenerateMetrics()
        {
            if (_LatestSample == null) _LatestSample = new GpuMetricSample();

            if (_UpdateFrequency.TotalSeconds <= 0)
            {
                Console.WriteLine("Ignoring GenerateMetrics: Update frequency must be greater than zero.");
                return;
            }

            var numUpdates = (int)(_RunLength.TotalSeconds / _UpdateFrequency.TotalSeconds);
            ClearCachedMetrics();

            for (var i = 0; i < numUpdates; i++)
            {
                Thread.Sleep(_UpdateFrequency);
                UpdateLatestMetrics();
            }
        }

        /// <summary>
        /// Clears the cached metric values in the watcher
        /// </summary>
        public void ClearCachedMetrics()
        {
            var result = _Watcher.GetLatest();
            for (var i = 0; i < MaxCacheClearAttempts; i++)
            {
                foreach (var res in result)
                {
                    _LatestSample.KeepMaxValues(res);
                    if (_LatestSample.IsValid()) return;
                }
                Thread.Sleep(_UpdateFrequency);
                result = _Watcher.GetLatest();
            }
            throw new TimeoutException("Cached metrics could not be cleared");
        }

        /// <summary>
        /// Updates the latest metrics sample based on the current metrics
        /// </summary>
        public void UpdateLatestMetrics()
        {
            var result = _Watcher.GetLatest();
            foreach (var res in result)
            {
                _LatestSample.KeepMaxValues(res);
            }
        }

        /// <summary>
        /// Updates the latest metrics sample's throughput value
        /// </summary>
        /// <param name="throughput">String representing the model throughput.</param>
        public void UpdateLatestMetricsThroughput(string throughput)
        {
            _LatestSample.Throughput = throughput;
        }

        /// <summary>
        /// Write metrics to filepath
        /// </summary>
        /// <param name="filePath">Filepath to write metrics to</param>
        public void ExportMetrics(string filepath)
        {
            using var writer = File.CreateText(filepath);

            writer.WriteLine(GpuMetricSample.GetHeader());

            foreach (var sample in _MetricsSamples)
                WriteSample(writer, sample);
        }

        /// <summary>
        /// Write metrics for single model to standard output
        /// </summary>
        public void ExportMetrics()
        {
            using var writer = new StreamWriter(Console.OpenStandardOutput())
            {
                AutoFlush = true
            };

            writer.WriteLine(GpuMetricSample.GetHeaderFormatted());

            foreach (var sample in _MetricsSamples)
                Console.WriteLine(sample.ToStringFormatted());
        }

        /// <summary>
        /// Prints the metrics in a sample
        /// </summary>
        /// <param name="writer">Writer to write values to.</param>
        /// <param name="sample">Sample to write.</param>
        private void WriteSample(StreamWriter writer, GpuMetricSample sample)
        {
            writer.WriteLine(sample.ToString());
        }
    }
}
