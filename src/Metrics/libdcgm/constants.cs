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

namespace Triton.MemoryAnalyzer.Metrics
{
    partial class libdcgm
    {
        public const int GroupMaxEntities = 64;
        public const int HealthWatchCount_v1 = 10;
        public const int MaxBlobLength = 4096;
        public const int MaxStringLength = 256;
        public const int MaxNumDevices = 16;
        public const int MaxFieldIdsPerFieldGroup = 128;


        // Identifies for special DCGM groups
        public const int GroupAllGpus = 0x7fffffff;
        public const int GroupAllNvSwitches = 0x7ffffffe;

        public const string LibraryName = "libdcgm.so";
    }


    internal static class dcgm_fi_dev
    {
        public const int bar_1_free = 93;
        public const int bar_1_total = 90;
        public const int bar_1_used = 92;
        public const int fb_free = 251;
        public const int fb_total = 250;
        public const int fb_used = 252;
        public const int gpu_util = 203;
        public const int memcpy_util = 204;
    }

    internal static class dcgm_field
    {
        /// <summary>
        /// Blob of binary data representing a structure.
        /// </summary>
        public const byte type_binary = (byte)'b';

        /// <summary>
        /// 8-byte double precision.
        /// </summary>
        public const byte type_double = (byte)'d';

        /// <summary>
        /// 8-byte signed integer.
        /// </summary>
        public const byte type_int64 = (byte)'i';

        /// <summary>
        /// Null-terminated ASCII Character string.
        /// </summary>
        public const byte type_string = (byte)'s';

        /// <summary>
        /// 8-byte signed integer usec since 1970.
        /// </summary>
        public const byte type_timestamp = (byte)'t';
    }
}
