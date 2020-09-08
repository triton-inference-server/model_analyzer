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
using System.Collections.Generic;
using System.Text;
using static System.Threading.Interlocked;

namespace Triton.MemoryAnalyzer.Metrics
{
    /// <summary>
    /// Represents field data values.
    /// </summary>
    public struct GpuMetricsData
    {
        public long MemoryUtilization { get; set; }
        public long GpuUtilization { get; set; }
        public long FreeBar1 { get; set; }
        public long UsedBar1 { get; set; }
        public long FreeFrameBuffer { get; set; }
        public long UsedFrameBuffer { get; set; }
    }

    /// <summary>
    /// Represents metrics data associated with a GPU.
    /// </summary>
    public struct LatestGpuMetrics
    {
        public int DeviceId { get; set; }
        public GpuMetricsData Data { get; set; }
    }

    /// <summary>
    /// Metrics watcher
    /// </summary>
    public interface IMetricsWatcher : IDisposable
    {
        /// <summary>
        /// GPU group under watch.
        /// </summary>
        IGpuGroup GpuGroup { get; }

        /// <summary>
        /// Get latest GPU usage metrics.
        /// </summary>
        IList<LatestGpuMetrics> GetLatest();
    }

    internal unsafe class Watcher : IMetricsWatcher
    {
        private const int StringBufferSize = 64 * 1024;
        private static readonly ushort[] DefaultFieldIds = new ushort[]
        {
            dcgm_fi_dev.memcpy_util,
            dcgm_fi_dev.gpu_util,
            dcgm_fi_dev.bar_1_free,
            dcgm_fi_dev.bar_1_used,
            dcgm_fi_dev.fb_free,
            dcgm_fi_dev.fb_used,
        };
        private static readonly UTF8Encoding Utf8 = new UTF8Encoding(false);

        private IntPtr _dcgmHandle;
        private readonly IGpuGroup _gpuGroup;
        private IntPtr _gpuGroupId;
        private readonly FieldGroup _gpuMetricsFieldGroup;
        private IntPtr _gpuMetricsGroupId;
        private readonly List<LatestGpuMetrics> _latestMetrics;
        private readonly object _syncpoint;
        private readonly System.Timers.Timer _pollTimer;


        /// <summary>
        /// Gpu metrics watcher.
        /// <para/>
        /// Initializes watcher and starts recording updates for metrics collection. If no device identifiers are specified, all supported devices are watched.
        /// <para/>
        /// </summary>
        /// <param name="dcgmHandle">
        /// DCGM Handle.
        /// </param>
        /// <param name="name">
        /// Name of the watcher.
        /// </param>
        /// <param name="updateFrequency">
        /// How often to update metrics in millisecond.
        /// </param>
        /// <param name="deviceIds">
        /// Device identifiers of GPUs to watch.
        /// </param>
        internal Watcher(IntPtr dcgmHandle, string name, TimeSpan updateFrequency, params int[] deviceIds)
        {
            if (dcgmHandle == IntPtr.Zero)
                throw new ArgumentNullException(nameof(dcgmHandle));
            if (name is null)
                throw new ArgumentNullException(nameof(name));
            if (deviceIds is null)
                throw new ArgumentNullException(nameof(deviceIds));

            _syncpoint = new object();
            _latestMetrics = new List<LatestGpuMetrics>();
            _dcgmHandle = dcgmHandle;

            // Generate new guid for this watcher
            var watcherId = Guid.NewGuid().ToString();

            // Create GPU Group
            var groupName = name + "_gpu_group_" + watcherId;
            var groupId = IntPtr.Zero;
            fixed (char* c = groupName)
            {
                var string_buffer = stackalloc byte[StringBufferSize];
                var len = Utf8.GetBytes(c, groupName.Length, string_buffer, StringBufferSize-1);

                string_buffer[len] = 0;

                if (deviceIds.Length == 0)
                {
                    var result = libdcgm.dcgmGroupCreate(dcgmHandle, dcgm_group_type.Default, string_buffer, &groupId);
                    if (result != dcgm_return.Ok)
                    {
                        throw new InvalidOperationException($"Error creating group. {Utils.errorString(result)}.");
                    }
                }
                else
                {
                    var result = libdcgm.dcgmGroupCreate(dcgmHandle, dcgm_group_type.Empty, string_buffer, &groupId);
                    if (result != dcgm_return.Ok)
                    {
                        throw new InvalidOperationException($"Error creating group. {Utils.errorString(result)}.");
                    }
                    foreach (uint id in deviceIds)
                    {
                        result = libdcgm.dcgmGroupAddDevice(dcgmHandle, groupId, id);
                        if (result != dcgm_return.Ok)
                        {
                            throw new InvalidOperationException($"Error adding gpu id {id} to group. {Utils.errorString(result)}.");
                        }
                    }
                }

            }

            // Instantiate GPU group
            _gpuGroupId = groupId;
            _gpuGroup = new GpuGroup(_dcgmHandle, _gpuGroupId);

            // Create GPU metrics field group
            var gpuMetricsGroupName = name + "_field_group_" + watcherId;
            var fieldIds = DefaultFieldIds;
            var fieldGroupId = IntPtr.Zero;
            fixed (ushort* f = fieldIds)
            {
                fixed (char* c = gpuMetricsGroupName)
                {
                    byte* string_buffer = stackalloc byte[StringBufferSize];
                    int len = Utf8.GetBytes(c, gpuMetricsGroupName.Length, string_buffer, StringBufferSize-1);

                    string_buffer[len] = 0;

                    dcgm_return result = libdcgm.dcgmFieldGroupCreate(_dcgmHandle, fieldIds.Length, f, string_buffer, &fieldGroupId);
                    if (result != dcgm_return.Ok)
                    {
                        throw new InvalidOperationException($"Error creating field group. {Utils.errorString(result)}.");
                    }
                }
            }

            // Instantiate GPU metrics field group
            _gpuMetricsGroupId = fieldGroupId;
            _gpuMetricsFieldGroup = new FieldGroup(_dcgmHandle, _gpuMetricsGroupId);


            var status = libdcgm.dcgmWatchFields(_dcgmHandle,
                                                 _gpuGroupId,
                                                 _gpuMetricsGroupId,
                                                 (ulong)updateFrequency.TotalMilliseconds*1000,
                                                 Math.Ceiling(updateFrequency.TotalSeconds),
                                                 0);

            if (status != dcgm_return.Ok)
            {
                throw new InvalidOperationException($"Error setting watches. {Utils.errorString(status)}.");
            }

            _pollTimer = new System.Timers.Timer(updateFrequency.TotalMilliseconds);
            _pollTimer.Elapsed += (object source, System.Timers.ElapsedEventArgs e) => GetLatestInternal();
            _pollTimer.AutoReset = true;
            _pollTimer.Enabled = true;

            GetLatestInternal();
        }

        ~Watcher()
        {
            Dispose(false);
        }

        protected void Dispose(bool disposing)
        {
            if (disposing)
            {
                // Stop polling dcgm host engine for values
                _pollTimer.Stop();
                _pollTimer.Dispose();

                // Fields are not being unwatched here because of the following dcgm limitation:
                // Having two sets of watches doesn't track two sets of data, so it's not a use case dcgm supports.
                // In embedded mode, dcgm only has one watch per fieldId per GPU.

                Exchange(ref _dcgmHandle, IntPtr.Zero);
                Exchange(ref _gpuGroupId, IntPtr.Zero);
                Exchange(ref _gpuMetricsGroupId, IntPtr.Zero);

                _gpuMetricsFieldGroup?.Dispose();
                _gpuGroup?.Dispose();
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public IList<LatestGpuMetrics> GetLatest()
        {
            LatestGpuMetrics[] metrics;

            lock (_syncpoint)
            {
                metrics = new LatestGpuMetrics[_latestMetrics.Count];
                _latestMetrics.CopyTo(metrics, 0);
            }

            return metrics;
        }

        private void GetLatestInternal()
        {
            if (CompareExchange(ref _dcgmHandle, IntPtr.Zero, IntPtr.Zero) == IntPtr.Zero)
                throw new InvalidOperationException("Failed operation because " + nameof(Metrics) + " has not been initialized.");

            ushort[] fieldIds = _gpuMetricsFieldGroup.GetInfo().FieldIds;
            var groupInfo = _gpuGroup.GetInfo();

            // Update dcgm fields
            libdcgm.dcgmUpdateAllFields(_dcgmHandle, 0);

            var latest = new List<LatestGpuMetrics>();

            fixed(ushort* pFieldIds = fieldIds)
            {
                foreach (var gpuId in groupInfo.DeviceIds)
                {
                    var gpuMetrics = new LatestGpuMetrics { DeviceId = gpuId };
                    dcgm_field_value_v1* values = stackalloc dcgm_field_value_v1[fieldIds.Length];

                    var status = libdcgm.dcgmGetLatestValuesForFields(_dcgmHandle,
                                                                      (uint)gpuId,
                                                                      pFieldIds,
                                                                      (uint)fieldIds.Length,
                                                                      values);
                    if (status != dcgm_return.Ok)
                    {
                        throw new InvalidOperationException($"Error fetching latest values for watches. {Utils.errorString(status)}.");
                    }

                    // Store fetched values in GpuMetrics struct
                    gpuMetrics.Data = new GpuMetricsData
                    {
                        MemoryUtilization = values[0].i64,
                        GpuUtilization = values[1].i64,
                        FreeBar1 = values[2].i64,
                        UsedBar1 = values[3].i64,
                        FreeFrameBuffer = values[4].i64,
                        UsedFrameBuffer = values[5].i64,
                    };

                    latest.Add(gpuMetrics);
                }
            }

            lock (_syncpoint)
            {
                _latestMetrics.Clear();
                _latestMetrics.AddRange(latest);
            }
        }

        public IGpuGroup GpuGroup
        {
            get { return _gpuGroup; }
        }
    }
}
