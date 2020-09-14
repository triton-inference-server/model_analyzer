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

using System;
using Triton.MemoryAnalyzer.Metrics;

namespace Triton.MemoryAnalyzer
{
    /// <summary>
    /// Class to store an instance of Gpu Metrics
    /// </summary>
    class GpuMetricSample
    {
        /// <summary>
        /// Model Batch Size
        /// </summary>
        public int BatchSize { get; }

        /// <summary>
        /// Concurrency Value
        /// </summary>
        public int ConcurrencyValue { get; }

        /// <summary>
        /// Maximum BAR1 memory, used for mapping framebuffer to CPU and third-party devices
        /// </summary>
        private long _MaxBar1 = 0;

        /// <summary>
        /// Maximum framebuffer memory
        /// </summary>
        private long _MaxFrameBuffer = 0;

        /// <summary>
        /// Maximum percentage of time during which one or more kernels was executing on the GPU
        /// </summary>
        private long _MaxGpuUtilization = 0;

        /// <summary>
        /// Maximum percentage of time during which global (device) memory was being read or written
        /// </summary>
        private long _MaxMemoryUtilization = 0;

        /// <summary>
        /// Name of the model associated with these metrics
        /// </summary>
        public string ModelName { get; } = "";

        /// <summary>
        /// Number of inference requests sent per second
        /// </summary>
        public string Throughput { get; set; } = "";

        /// <summary>
        /// Constructor for GpuMetricSample
        /// </summary>
        /// <param name="modelName">Name of model associated with sample.</param>
        /// <param name="batchSize">Number of models sent per batch to server.</param>
        /// <param name="concurrencyValue">Number of concurrent threads used by server.</param>
        /// <returns>Returns new sample.</returns>
        public GpuMetricSample(string modelName, int batchSize, int concurrencySize)
        {
            BatchSize = batchSize;
            ConcurrencyValue = concurrencySize;
            ModelName = modelName;
        }

        /// <summary>
        /// Constructor for GpuMetricSample
        /// </summary>
        public GpuMetricSample()
        {

        }

        /// <summary>
        /// Compare current values against a LatestGpuMetrics object and keep max values
        /// </summary>
        /// <param name="metrics">Metrics to compare with.</param>
        public void KeepMaxValues(LatestGpuMetrics metrics)
        {
            _MaxMemoryUtilization = Math.Max(_MaxMemoryUtilization, metrics.Data.MemoryUtilization);
            _MaxGpuUtilization = Math.Max(_MaxMemoryUtilization, metrics.Data.GpuUtilization);
            _MaxBar1 = Math.Max(_MaxMemoryUtilization, metrics.Data.UsedBar1);
            _MaxFrameBuffer = Math.Max(_MaxMemoryUtilization, metrics.Data.UsedFrameBuffer);
        }

        /// <summary>
        /// Return header with metric column names
        /// </summary>
        /// <returns>Returns comma-delimited header.</returns>
        public static string GetHeader()
        {
            return "Model,Batch,Concurrency,Throughput,MaxMemoryUtil(%),MaxGpuUtil(%)," +
                "_MaxBar1(MB),MaxFramebuffer(MB)";
        }

        /// <summary>
        /// Return header with metric column names
        /// </summary>
        /// <returns>Returns comma-delimited header.</returns>
        public static string GetHeaderFormatted()
        {
            return string.Format
                (
                "{0,-30}{1,-20}{2,-20}{3,-20}{4,-20}{5,-20}{6,-20}{7,-20}",
                "Model",
                "Batch",
                "Concurrency",
                "Throughput",
                "Max Memory Util(%)",
                "Max GPU Util(%) ",
                "Max BAR1(MB)",
                "Max Framebuffer(MB)"
                );
        }

        /// <summary>
        /// Return field values
        /// </summary>
        /// <returns>Returns comma-delimited metric values.</returns>
        public override string ToString()
        {
            return $"{ModelName},{BatchSize},{ConcurrencyValue},{Throughput},{_MaxMemoryUtilization},{_MaxGpuUtilization}," +
                $"{_MaxBar1},{_MaxFrameBuffer}";
        }

        /// <summary>
        /// Return field values formatted for standard output
        /// </summary>
        /// <param name="modelName">Name of model.</param>
        /// <param name="batchSize">Number of models sent concurrently to server.</param>
        /// <returns>Returns comma-delimited metric values.</returns>
        public string ToStringFormatted()
        {
            return string.Format
                (
                $"{ModelName,-30}" +
                $"{BatchSize,-20}" +
                $"{ConcurrencyValue,-20}" +
                $"{Throughput,-20}" +
                $"{_MaxMemoryUtilization,-20}" +
                $"{_MaxGpuUtilization,-20}" +
                $"{_MaxBar1,-20}" +
                $"{_MaxFrameBuffer,-20}"
                );
        }

        /// <summary>
        /// Checks whether the sample is valid, to validate the values are not from DCGM's buffer
        /// </summary>
        /// <returns>Returns true if valid, else false</returns>
        public bool IsValid()
        {
            return (_MaxFrameBuffer != 0);
        }
    }
}
