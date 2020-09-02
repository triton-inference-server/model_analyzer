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
using System.Diagnostics;
using System.Xml;
using Xunit;

namespace ModelAnalyzer.Metrics
{
    public class GpuMetricsTest
    {
        public static readonly TimeSpan UpdateFrequencyMaximum = TimeSpan.FromSeconds(60);
        public static readonly TimeSpan UpdateFrequencyMinimum = TimeSpan.FromMilliseconds(50);

        [Fact]
        public void GetAllSupportedGpusTest()
        {
            var gpuMetrics = new GpuMetrics();

            var gpuIdList1 = gpuMetrics.GetAllSupportedGpus();
            var gpuIdList2 = gpuMetrics.GetAllSupportedGpus();
            var gpuIdList3 = gpuMetrics.GetAllSupportedGpus();

            // Verify consistency between different calls to GpuMetrics.GetAllSupportedGpus()
            Assert.Equal(gpuIdList1, gpuIdList2);
            Assert.Equal(gpuIdList2, gpuIdList3);

            var startInfo = new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "-q -x",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                WorkingDirectory = Environment.CurrentDirectory,
                CreateNoWindow = false,
            };

            /*
            This test is only valid if all GPUs on the system are supported by DCGM (Data Center GPU Manager).
            For more details on the supported GPUs please refer DCGM team.
            */
            using var process = new Process() { StartInfo = startInfo };
            process.Start();
            string nvidiaSmiXml = process.StandardOutput.ReadToEnd();
            var doc = new XmlDocument();
            doc.LoadXml(nvidiaSmiXml);
            var expectedNumberOfGpus = doc.DocumentElement.SelectSingleNode("/nvidia_smi_log/attached_gpus")?.InnerText;

            // Verify number of GPUs reported by GpuMetrics.GetAllSupportedGpus() and nvidia-smi are same.
            Assert.Equal(expectedNumberOfGpus, gpuIdList1.Length.ToString());
        }

        [Fact]
        public void WatcherWithDefaultGpuGroupTest()
        {
            var gpuMetrics = new GpuMetrics();
            var expectedGpuIdList = gpuMetrics.GetAllSupportedGpus();

            using var watcher = gpuMetrics.GetWatcher("metrics", TimeSpan.FromMilliseconds(100));
            var gpuGroup = watcher.GpuGroup;
            Assert.NotNull(gpuGroup);

            var groupInfo = gpuGroup.GetInfo();
            var deviceIds = groupInfo.DeviceIds;

            // Verify that deviceIds value is not null or empty when GpuGroup.GetInfo() is called.
            Assert.NotNull(deviceIds);
            Assert.NotEmpty(deviceIds);

            // Verify that GPU group contains all GPU identifiers.
            Assert.Equal(expectedGpuIdList, deviceIds);
        }

        [Fact]
        public void WatcherWithCustomGpuGroupTest()
        {
            var gpuMetrics = new GpuMetrics();
            var gpuIds = gpuMetrics.GetAllSupportedGpus();

            // Create a custom GPU group containing only half of the actual number of GPUs on the system.
            var lastIdx = (int) Math.Ceiling((double)gpuIds.Length/2);
            var expectedGpuIdList = gpuIds[0..lastIdx];

            // Verify that correct GPU group info is returned.
            using var watcher = gpuMetrics.GetWatcher("metrics", TimeSpan.FromMilliseconds(100), gpuIds[0..lastIdx]);
            var gpuGroup = watcher.GpuGroup;
            Assert.NotNull(gpuGroup);

            var groupInfo = gpuGroup.GetInfo();
            var deviceIds = groupInfo.DeviceIds;

            // Verify that device identifiers are not null or empty when GpuGroup.GetInfo() is called.
            Assert.NotNull(deviceIds);
            Assert.NotEmpty(deviceIds);

            // Verify that GPU group contains all expected device identifiers.
            Assert.Equal(expectedGpuIdList, deviceIds);
        }

        [Fact]
        public void GetLatestGpuMetricsTest()
        {
            var gpuMetrics = new GpuMetrics();
            var expectedGpuIdList = gpuMetrics.GetAllSupportedGpus();

            // 100 millisecond update frequency.
            using IMetricsWatcher watcher = gpuMetrics.GetWatcher("metrics", TimeSpan.FromMilliseconds(100));
            var gpuGroup = watcher.GpuGroup;

            // delay of 200 milliseconds before getting latest values.
            System.Threading.Thread.Sleep(200);

            // Verify if metrics result is not null.
            var result1 = watcher.GetLatest();
            Assert.NotNull(result1);

            // Verify if we result contains entries for all GPUs.
            int[] actualGpuIdList = new int[result1.Count];
            for (int i = 0; i < result1.Count; i++)
            {
                actualGpuIdList[i] = result1[i].DeviceId;
            }

            Assert.Equal(expectedGpuIdList, actualGpuIdList);

            // Verify if Metrics data values are valid.
            foreach (var metricsData in result1)
            {
                Assert.InRange(metricsData.Data.MemoryUtilization, 0, 100);
                Assert.InRange(metricsData.Data.GpuUtilization, 0, 100);
                Assert.True(metricsData.Data.FreeBar1 >= 0);
                Assert.True(metricsData.Data.UsedBar1 >= 0);
                Assert.True(metricsData.Data.FreeFrameBuffer >= 0);
                Assert.True(metricsData.Data.UsedFrameBuffer >= 0);
            }

            // Verify if memory values are consistent.
            // Irrespective of GPU usage between different calls, sum of free BAR1 and used BAR1 must be same.
            // Irrespective of GPU usage between different calls, sum of free frame buffers and used frame buffers must be same.
            var result2 = watcher.GetLatest();
            Assert.Equal(result1.Count, result2.Count);

            for (int i = 0; i < result1.Count && i < result2.Count; i++)
            {
                Assert.Equal(result1[i].Data.FreeBar1 + result1[i].Data.UsedBar1,
                             result2[i].Data.FreeBar1 + result2[i].Data.UsedBar1);
                Assert.Equal(result1[i].Data.FreeFrameBuffer + result1[i].Data.UsedFrameBuffer,
                             result2[i].Data.FreeFrameBuffer + result2[i].Data.UsedFrameBuffer);
            }
        }

        [Fact]
        public void UpdateFrequencyRangeTest()
        {
            var gpuMetrics = new GpuMetrics();
            var expectedGpuIdList = gpuMetrics.GetAllSupportedGpus();

            // Verify if exception is thrown when update frequency is set 1 millisecond less than its valid minimum value.
            Assert.Throws<ArgumentOutOfRangeException>(() => gpuMetrics.GetWatcher("metrics", UpdateFrequencyMinimum - TimeSpan.FromMilliseconds(1)));

            // Verify if exception is thrown when update frequency is set 1 millisecond greater than its valid maximum value.
            Assert.Throws<ArgumentOutOfRangeException>(() => gpuMetrics.GetWatcher("metrics", UpdateFrequencyMaximum + TimeSpan.FromMilliseconds(1)));
        }
    }
}