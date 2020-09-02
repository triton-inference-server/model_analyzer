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
using static System.Threading.Interlocked;

namespace ModelAnalyzer.Metrics
{
    /// <summary>
    /// GPU group information.
    /// </summary>
    public struct GroupInfo
    {
        public string GroupName { get; set; }

        public IReadOnlyList<int> DeviceIds { get; set; }
    }

    /// <summary>
    /// GPU group.
    /// </summary>
    public interface IGpuGroup : IDisposable
    {
        /// <summary>
        /// Gets the GPU Group Info
        /// </summary>
        GroupInfo GetInfo();
    }

    /// <summary>
    /// GPU group.
    /// </summary>
    internal unsafe class GpuGroup : IGpuGroup
    {
        private IntPtr _dcgmHandle;
        private IntPtr _groupId;

        /// <summary>
        /// Creates a GPU group.
        /// </summary>
        internal GpuGroup(IntPtr dcgmHandle, IntPtr gpuGroupId)
        {
            if(dcgmHandle == IntPtr.Zero)
                throw new ArgumentNullException(nameof(dcgmHandle));
            if(gpuGroupId == IntPtr.Zero)
                throw new ArgumentNullException(nameof(gpuGroupId));

            _dcgmHandle = dcgmHandle;
            _groupId = gpuGroupId;
        }

        ~GpuGroup()
        {
            Dispose();
        }

        public void Dispose()
        {
            IntPtr dcgmHandle;
            if((dcgmHandle = Exchange(ref _dcgmHandle, IntPtr.Zero)) != IntPtr.Zero)
            {
                IntPtr groupId;
                if ((groupId = Exchange(ref _groupId, IntPtr.Zero)) != IntPtr.Zero)
                {
                    libdcgm.dcgmGroupDestroy(dcgmHandle, groupId);
                }
            }
        }

        public GroupInfo GetInfo()
        {
            if (CompareExchange(ref _dcgmHandle, IntPtr.Zero, IntPtr.Zero) == IntPtr.Zero)
                throw new InvalidOperationException("Failed operation because " + nameof(Metrics) + " has not been initialized.");
            if (CompareExchange(ref _groupId, IntPtr.Zero, IntPtr.Zero) == IntPtr.Zero)
                throw new NullReferenceException("Failed operation because group identity has not been initialized.");

            dcgm_group_info_v2 dcgmGroupInfo = new dcgm_group_info_v2 { };
            dcgmGroupInfo.version = Utils.dcgmGroupInfoVersion();

            dcgm_return status = libdcgm.dcgmGroupGetInfo(_dcgmHandle, _groupId, &dcgmGroupInfo);
            if (status != dcgm_return.Ok)
            {
                throw new InvalidOperationException($"Error getting group information. {Utils.errorString(status)}.");
            }

            dcgm_group_entity_pair* entity = (dcgm_group_entity_pair*) dcgmGroupInfo.entityList;
            var entityIds = new List<int>();
            for (int i = 0; i < dcgmGroupInfo.count; i++)
            {
                entityIds.Add(entity[i].entity_id);
            }

            var groupInfo = new GroupInfo
            {
                GroupName = new string((sbyte*)dcgmGroupInfo.groupName),
                DeviceIds = entityIds,
            };

            return groupInfo;
        }
    }
}
