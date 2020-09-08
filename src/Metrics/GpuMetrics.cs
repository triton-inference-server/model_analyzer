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

namespace Triton.MemoryAnalyzer.Metrics
{
    /// <summary>
    /// GPU metrics interface
    /// </summary>
    public interface IGpuMetrics
    {
        /// <summary>
        /// Get identifiers corresponding to all the Metrics-supported devices on the system.
        /// </summary>
        int[] GetAllSupportedGpus();

        /// <summary>
        /// Gets a metrics watcher.
        /// <para/>
        /// Initializes watcher and starts recording updates for following metrics on all specified GPUs. If no GPU identifiers are specified, all GPUs are watched.
        /// <para/>
        /// Memory utilization, GPU utilization, free and used bar1 of GPU in MB and, free and used frame buffer in MB.
        /// <para/>
        /// NOTE: Use <see cref="GetAllSupportedGpus"/> to get identifiers corresponding to all the Metrics-supported devices on the system.
        /// <para/>
        /// </summary>
        /// <param name="name"> Name of the watcher. </param>
        /// <param name="updateFrequency"> How often to update metrics in millisecond. (Valid Range: 50 milliseconds to 60 seconds)</param>
        /// <param name="gpuGroup"> User defined GPU group. </param>
        IMetricsWatcher GetWatcher(string name, TimeSpan updateFrequency, params int[] deviceIds);
    }

    /// <summary>
    /// GPU metrics
    /// </summary>
    public unsafe class GpuMetrics : IGpuMetrics
    {
        private static readonly IntPtr _dcgmHandle;

        public Type ServiceType
            => typeof(IGpuMetrics);

        static GpuMetrics()
        {
            if (Environment.OSVersion.Platform == PlatformID.Win32NT)
                return;

            try
            {
                // Initialize DCGM within this process
                var status = libdcgm.dcgmInit();
                if (status != dcgm_return.Ok)
                    throw new InvalidOperationException($"Error initializing DCGM engine. {Utils.errorString(status)}.");

                var dcgmHandle = IntPtr.Zero;

                // Start Dcgm host engine
                status = libdcgm.dcgmStartEmbedded(dcgm_operation_mode.auto, &dcgmHandle);
                if (status != dcgm_return.Ok)
                    throw new InvalidOperationException($"Error starting embedded DCGM engine. {Utils.errorString(status)}.");

                _dcgmHandle = dcgmHandle;
            }
            catch
            {
                if (Environment.OSVersion.Platform == PlatformID.Unix)
                    throw;
            }
        }

        public int[] GetAllSupportedGpus()
        {
            var p = stackalloc uint[libdcgm.MaxNumDevices];
            var count = 0;

            var status = libdcgm.dcgmGetAllSupportedDevices(_dcgmHandle, p, &count);

            var deviceIdList = new int[count];
            fixed (int* pDeviceIdList = deviceIdList)
            {
                Buffer.MemoryCopy(p, pDeviceIdList, count * sizeof(int), count * sizeof(int));
            }

            return deviceIdList;
        }

        public IMetricsWatcher GetWatcher(string name, TimeSpan updateFrequency, params int[] deviceIds)
        {
            if (updateFrequency < UpdateFrequency.Minimum || updateFrequency > UpdateFrequency.Maximum)
                throw new ArgumentOutOfRangeException(nameof(updateFrequency));

            return new Watcher(_dcgmHandle, name, updateFrequency, deviceIds);
        }
    }

    public static class UpdateFrequency
    {
        public static readonly TimeSpan Maximum = TimeSpan.FromSeconds(60);
        public static readonly TimeSpan Minimum = TimeSpan.FromMilliseconds(50);
    }
}
