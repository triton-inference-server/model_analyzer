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
using static System.Threading.Interlocked;

namespace Triton.MemoryAnalyzer.Metrics
{
    public struct FieldGroupInfo
    {
        public string GroupName { get; set; }

        public ushort[] FieldIds { get; set; }
    }

    /// <summary>
    /// Field group.
    /// </summary>
    public interface IFieldGroup : IDisposable
    {
        /// <summary>
        /// Gets the field group information.
        /// </summary>
        FieldGroupInfo GetInfo();
    }

    /// <summary>
    /// Field group.
    /// </summary>
    internal unsafe class FieldGroup : IFieldGroup
    {
        private IntPtr _dcgmHandle;
        private IntPtr _fieldGroupId;

        // Create Field Group
        internal FieldGroup(IntPtr dcgmHandle, IntPtr fieldGroupId)
        {
            if(dcgmHandle == IntPtr.Zero)
                throw new ArgumentNullException(nameof(dcgmHandle));
            if(fieldGroupId == IntPtr.Zero)
                throw new ArgumentNullException(nameof(fieldGroupId));

            _dcgmHandle = dcgmHandle;
            _fieldGroupId = fieldGroupId;
        }

        ~FieldGroup()
        {
            Dispose();
        }

        public void Dispose()
        {
            IntPtr dcgmHandle;
            if((dcgmHandle = Exchange(ref _dcgmHandle, IntPtr.Zero)) != IntPtr.Zero)
            {
                IntPtr fieldGroupId;
                if ((fieldGroupId = Exchange(ref _fieldGroupId, IntPtr.Zero)) != IntPtr.Zero)
                {
                    libdcgm.dcgmFieldGroupDestroy(dcgmHandle, fieldGroupId);
                }
            }
        }

        public FieldGroupInfo GetInfo()
        {
            if (CompareExchange(ref _dcgmHandle, IntPtr.Zero, IntPtr.Zero) == IntPtr.Zero)
                throw new InvalidOperationException("Failed operation because " + nameof(Metrics) + " has not been initialized.");
            if (CompareExchange(ref _fieldGroupId, IntPtr.Zero, IntPtr.Zero) == IntPtr.Zero)
                throw new NullReferenceException("Failed operation because field group identity has not been initialized.");

            var dcgmFieldGroupInfo = new dcgm_field_group_info_v1
            {
                version = Utils.dcgmFieldGroupInfoVersion1(),
                fieldGroupId = _fieldGroupId,
            };

            var result = libdcgm.dcgmFieldGroupGetInfo(_dcgmHandle, &dcgmFieldGroupInfo);
            if (result != dcgm_return.Ok)
            {
                throw new InvalidOperationException($"Error creating field group. {Utils.errorString(result)}.");
            }

            var fieldGroupInfo = new FieldGroupInfo
            {
                GroupName = new string((sbyte*)dcgmFieldGroupInfo.fieldGroupName),
                FieldIds = new ushort[dcgmFieldGroupInfo.numFieldIds],
            };

            for (var i = 0; i < dcgmFieldGroupInfo.numFieldIds; i++)
            {
                fieldGroupInfo.FieldIds[i] = dcgmFieldGroupInfo.fieldIds[i];
            }

            return fieldGroupInfo;
        }
    }
}
