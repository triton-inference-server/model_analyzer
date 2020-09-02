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
using System.Runtime.InteropServices;

namespace ModelAnalyzer.Metrics
{
    /// <summary>
    /// Represents a set of memory, SM, and video clocks for a device. This can be current values or a target values based on context
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct dcgm_clock_set_v1
    {
        /// <summary>
        /// Version Number (<seealso cref="dcgmClockSet_version"/>)
        /// </summary>
        public int version;

        /// <summary>
        /// Memory Clock
        /// <para/>
        /// Memory Clock value OR DCGM_INT32_BLANK to Ignore/Use compatible value with <seealso cref="sm_clock"/>.
        /// </summary>
        public uint mem_clock;

        /// <summary>
        /// SM Clock
        /// <para/>
        /// SM Clock value OR DCGM_INT32_BLANK to Ignore/Use compatible value with <seealso cref="mem_clock"/>.
        /// </summary>
        public uint sm_clock;
    }

    /// <summary>
    /// Used to represent Performance state settings
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct dcgm_config_perf_state_settings
    {
        /// <summary>
        /// Sync Boost Mode.
        /// <list type="1">
        ///  <item>0 : Disabled</item>
        ///  <item>1 : Enabled</item>
        ///  <item>DCGM_INT32_BLANK : Ignored</item>
        /// </list>
        /// Note that using this setting may result in lower clocks than targetClocks
        /// </summary>
        public uint sync_boost;

        /// <summary>
        /// Target clocks.
        /// <para/>
        /// Set smClock and memClock to <see cref="DCGM_INT32_BLANK"/> to ignore/use compatible values.
        /// <para/>
        /// For GPUs > Maxwell, setting this implies autoBoost=0
        /// </summary>
        public dcgm_clock_set_v1 target_clocks;
    }

    /// <summary>
    /// Represents the power cap for each member of the group.
    /// </summary>
    internal enum dcgm_config_power
    {
        /// <summary>
        /// Represents the power cap to be applied for each member of the group
        /// </summary>
        cap_individual = 0,

        /// <summary>
        /// Represents the power budget for the entire group
        /// </summary>
        budget_group = 1
    }

    /// <summary>
    /// Used to represents the power capping limit for each GPU in the group or to represent the power 
    /// budget for the entire group
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct dcgm_config_power_limit
    {
        /// <summary>
        /// Flag to represent power cap for each GPU or power budget for the group of GPUs.
        /// </summary>
        public dcgm_config_power type;

        /// <summary>
        /// Power Limit in Watts (Set a value OR DCGM_INT32_BLANK to Ignore).
        /// </summary>
        public uint val;
    }

    /// <summary>
    /// Structure to represent default and target configuration for a device
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct dcgm_config_v1
    {
        /// <summary>
        /// Version number (<seealso cref="dcgmConfig_version"/>)
        /// </summary>
        public uint version;

        /// <summary>
        /// GPU Identifier
        /// </summary>
        public uint gpu_id;

        /// <summary>
        /// ECC Mode
        /// <list type="bullet">
        ///  <item>0 : Disabled</item>
        ///  <item>1 : Enabled</item>
        ///  <item>DCGM_INT32_BLANK : Ignored</item>
        /// </list>
        /// </summary>
        public uint ecc_mode;

        /// <summary>
        /// Compute Mode (One of DCGM_CONFIG_COMPUTEMODE_? OR DCGM_INT32_BLANK to Ignore)
        /// </summary>
        public uint compute_mode;

        /// <summary>
        /// Performance State Settings (clocks / boost mode)
        /// </summary>
        public dcgm_config_perf_state_settings perf_state;

        /// <summary>
        /// Power Limits.
        /// </summary>
        public dcgm_config_power_limit power_limit;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct dcgm_device_vgpu_util_info_v1
    {
        public uint dec_util;

        public uint enc_util;

        public uint mem_util;

        public uint sm_util;

        /// <summary>
        /// Version Number (dcgmDeviceVgpuUtilInfo_version).
        /// </summary>
        public uint version;

        /// <summary>
        /// vGPU instance Identifier
        /// </summary>
        public uint vgpu_id;
    }

    /// <summary>
    /// Enum of possible field entity groups
    /// </summary>
    internal enum dcgm_field_entity_group : int
    {
        /// <summary>
        /// Field is not associated with an entity.
        /// </summary>
        None = 0,

        /// <summary>
        /// Field is associated with a GPU entity.
        /// </summary>
        Gpu,

        /// <summary>
        /// Field is associated with a VGPU entity.
        /// </summary>
        Vgpu,

        /// <summary>
        /// Field is associated with a Switch entity
        /// </summary>
        Switch,

        /// <summary>
        /// Number of elements in this enumeration.
        /// </summary>
        Count // Keep this entry last.
    }


    /// <summary>
    /// Structure to store DCGM field group information.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct dcgm_field_group_info_v1
    {
        /// <summary>
        /// Version Number
        /// </summary>
        public uint version;

        /// <summary>
        /// Number of entries in fieldIds[] that are valid
        /// </summary>
        public uint numFieldIds;

        /// <summary>
        /// Field Group Identifier
        /// </summary>
        public IntPtr fieldGroupId;

        /// <summary>
        /// Field Group Name.
        /// </summary>
        public fixed byte fieldGroupName[libdcgm.MaxStringLength];

        /// <summary>
        /// List of the fieldIds that are in this group
        /// </summary>
        public fixed ushort fieldIds[libdcgm.MaxFieldIdsPerFieldGroup];
    }


    /// <summary>
    /// Structure to store meta data for the field
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct dcgm_field_meta
    {
        /// <summary>
        /// Field identifier.
        /// </summary>
        public ushort field_id;

        /// <summary>
        /// Field type. DCGM_FT_? #define
        /// </summary>
        public byte field_type;

        /// <summary>
        /// Field size in bytes (raw value size).
        /// <para/>
        /// 0=variable
        /// </summary>
        public byte size;

        /// <summary>
        /// Tag for this field for serialization like 'device_temperature'.
        /// </summary>
        public fixed byte tag[48];

        /// <summary>
        /// Field scope.
        /// </summary>
        public int scope;

        /// <summary>
        /// Optional NVML field this DCGM field maps to. 0 = no mapping.
        /// <para/>
        /// Otherwise, this should be a NVML_FI_? #define from nvml.h
        /// </summary>
        public int nvml_field_id;

        /// <summary>
        /// Pointer to the structure that holds the formatting the values for fields.
        /// </summary>
        public dcgm_field_output_format* value_format;
    }

    /// <summary>
    /// Structure for formating the output for dmon. Used as a member in dcgm_field_meta
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct dcgm_field_output_format
    {
        /// <summary>
        /// Short name corresponding to field.
        /// <para/>
        /// This short name is used to identify columns in dmon output.
        /// </summary>
        public fixed byte short_name[10];

        /// <summary>
        /// The unit of value. Eg: C(elsius), W(att), MB/s short width;
        /// <para/>
        /// Maximum width/number of digits that a value for field can have.
        /// </summary>
        public fixed byte unit[4];
    }

    [StructLayout(LayoutKind.Explicit)]
    internal unsafe struct dcgm_field_value_v1
    {
        [FieldOffset(0)]
        public uint version;

        [FieldOffset(4)]
        public ushort field_id;

        [FieldOffset(8)]
        public ushort field_type;

        [FieldOffset(12)]
        public int status;

        [FieldOffset(16)]
        public long ts;

        [FieldOffset(24)]
        public long i64;

        [FieldOffset(24)]
        public double dbl;

        [FieldOffset(24)]
        public fixed byte str[libdcgm.MaxStringLength];

        [FieldOffset(24)]
        public fixed byte blob[libdcgm.MaxBlobLength];
    }

    /// <summary>
    /// Represents a entityGroupId + entityId pair to uniquely identify a given entityId inside
    /// a group of entities
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Size = dcgm_group_entity_pair.Size)]
    internal struct dcgm_group_entity_pair
    {
        public const int Size = 8;

        /// <summary>
        /// Entity Group Identifier entity belongs to.
        /// </summary>
        public dcgm_field_entity_group entity_group_id;

        /// <summary>
        /// Entity Identifier of the entity
        /// </summary>
        public int entity_id;
    }

    /// <summary>
    /// Structure to store information for DCGM group
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct dcgm_group_info_v2
    {
        /// <summary>
        /// Version Number
        /// </summary>
        public uint version;

        /// <summary>
        /// count of entityIds returned in entityList.
        /// </summary>
        public uint count;

        /// <summary>
        /// Group Name.
        /// </summary>
        public fixed byte groupName[libdcgm.MaxStringLength];

        /// <summary>
        /// List of the entities that are in this group
        /// </summary>
        // Array of entities of type dcgm_group_entity_pair
        public fixed byte entityList[libdcgm.GroupMaxEntities * dcgm_group_entity_pair.Size];
    }

    /// <summary>
    /// Type of GPU groups
    /// </summary>
    internal enum dcgm_group_type : int
    {
        /// <summary>
        /// All the GPUs on the node are added to the group
        /// </summary>
        Default = 0,

        /// <summary>
        /// Creates an empty group
        /// </summary>
        Empty = 1,

        /// <summary>
        /// All NvSwitches of the node are added to the group
        /// </summary>
        NvSwitches = 2,
    };

    /// <summary>
    /// Health Response structure version 1. GPU Only
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct dcgm_health_response_v1
    {
        /// <summary>
        /// Version number. (<seealso cref="dcgm_health_response_version"/>)
        /// </summary>
        public uint version;

        /// <summary>
        /// The overall health of the system. (<seealso cref="dcgm_health_watch_results"/>)
        /// </summary>
        public dcgm_health_watch_results overall_health;

        /// <summary>
        /// The number of GPUs with warnings/errors.
        /// </summary>
        public uint gpu_count;

        public fixed byte gpu[libdcgm.MaxNumDevices * dcgm_health_response_v1_gpu.Size];
    }

    [StructLayout(LayoutKind.Sequential, Size = dcgm_health_response_v1_gpu.Size)]
    internal unsafe struct dcgm_health_response_v1_gpu
    {
        public const int Size = libdcgm.HealthWatchCount_v1 * dcgm_health_response_v1_system.Size + 12;

        /// <summary>
        /// GPU Identifier for which this data is valid.
        /// </summary>
        public uint gpu_id;

        /// <summary>
        /// Overall health of this GPU.
        /// </summary>
        public dcgm_health_watch_results overall_health;

        /// <summary>
        /// The number of systems that encountered a warning/error.
        /// </summary>
        public uint incident_count;

        public fixed byte systems[libdcgm.HealthWatchCount_v1 * dcgm_health_response_v1_system.Size];
    }

    [StructLayout(LayoutKind.Sequential, Size = dcgm_health_response_v1_system.Size)]
    internal unsafe struct dcgm_health_response_v1_system
    {
        public const int Size = 1032;

        /// <summary>
        /// System to which this information belongs.
        /// </summary>
        public dcgm_health_systems_watch system;

        /// <summary>
        /// Health of the specified system on this GPU.
        /// </summary>
        public dcgm_health_watch_results health;

        /// <summary>
        /// Information about the error(s) or warning(s) flagged.
        /// </summary>
        public fixed byte error_string[1024];
    }

    /// <summary>
    /// Systems structure used to enable or disable health watch systems
    /// </summary>
    [Flags]
    internal enum dcgm_health_systems_watch : uint
    {
        /// <summary>
        /// PCIE System Watches (Must Have 1M Of Data Before Query)
        /// </summary>
        pcie = 0x1,

        /// <summary>
        /// Nvlink System Watches
        /// </summary>
        nvlink = 0x2,

        /// <summary>
        /// Power Management Unit Watches
        /// </summary>
        pmu = 0x4,

        /// <summary>
        /// Microcontroller Unit Watches
        /// </summary>
        mcu = 0x8,

        /// <summary>
        /// Memory Watches
        /// </summary>
        mem = 0x10,

        /// <summary>
        /// Streaming Multiprocessor Watches
        /// </summary>
        sm = 0x20,

        /// <summary>
        /// Inforom Watches
        /// </summary>
        inforom = 0x40,

        /// <summary>
        /// Temperature Watches (Must Have 1M Of Data Before Query)
        /// </summary>
        thermal = 0x80,

        /// <summary>
        /// Power Watches (Must Have 1M Of Data Before Query)
        /// </summary>
        power = 0x100,

        /// <summary>
        /// Driver-Related Watches
        /// </summary>
        driver = 0x200,

        /// <summary>
        /// Non-Fatal Errors In Nvswitch
        /// </summary>
        nvswitch_non_fatal = 0x400,

        /// <summary>
        /// Fatal Errors In Nvswitch
        /// </summary>
        nvsiwtch_fatal = 0x800,

        /// <summary>
        /// All watches enabled
        /// </summary>
        all = 0xFFFFFFFF,
    }

    /// <summary>
    /// Health Watch test results
    /// </summary>
    internal enum dcgm_health_watch_results
    {
        /// <summary>
        /// All results within this system are reporting normal.
        /// </summary>
        pass = 0,

        /// <summary>
        /// A warning has been issued, refer to the response for more information.
        /// </summary>
        warn = 10,

        /// <summary>
        /// A failure has been issued, refer to the response for more information.
        /// </summary>
        fail = 20,
    }

    /// <summary>
    /// Operation mode for DCGM
    /// <para/>
    /// DCGM can run in auto-mode where it runs additional threads in the background to collect
    /// any metrics of interest and auto manages any operations needed for policy management.
    /// <para/>
    /// DCGM can also operate in manual-mode where it's execution is controlled by the user. In
    /// this mode, the user has to periodically call APIs such as dcgmPolicyTrigger and
    /// <see cref="dcgm.dcgmUpdateAllFields"/> which tells DCGM to wake up and perform data collection and
    /// operations needed for policy management.
    /// </summary>
    public enum dcgm_operation_mode : int
    {
        unknown = 0,

        auto = 1,

        manual = 2,
    };

    internal enum dcgm_return
    {
        /// <summary>
        /// Success
        /// </summary>
        Ok = 0,

        /// <summary>
        /// A Bad Parameter Was Passed To A Function
        /// </summary>
        BadParam = -1,

        /// <summary>
        /// A Generic, Unspecified Error
        /// </summary>
        GenericError = -3,

        /// <summary>
        /// An Out Of Memory Error Occurred
        /// </summary>
        Memory = -4,

        /// <summary>
        /// Setting Not Configured
        /// </summary>
        NotConfigured = -5,

        /// <summary>
        /// Feature Not Supported
        /// </summary>
        NotSupported = -6,

        /// <summary>
        /// Dcgm Init Error
        /// </summary>
        InitError = -7,

        /// <summary>
        /// When Nvml Returns Error
        /// </summary>
        NvmlError = -8,

        /// <summary>
        /// Object Is In Pending State Of Something Else
        /// </summary>
        Pending = -9,

        /// <summary>
        /// Object Is In Undefined State
        /// </summary>
        Uninitialized = -10,

        /// <summary>
        /// Requested Operation Timed Out
        /// </summary>
        Timeout = -11,

        /// <summary>
        /// Version Mismatch Between Received And Understood Api
        /// </summary>
        VersionMismatch = -12,

        /// <summary>
        /// Unknown Field Identifier
        /// </summary>
        UnknownField = -13,

        /// <summary>
        /// No Data Is Available
        /// </summary>
        NoData = -14,

        /// <summary>
        /// Data Is Considered Stale
        /// </summary>
        StaleData = -15,

        /// <summary>
        /// The Given Field Identifier Is Not Being Updated By The Cache Manager
        /// </summary>
        NotWatched = -16,

        /// <summary>
        /// Do Not Have Permission To Perform The Desired Action
        /// </summary>
        NoPermission = -17,

        /// <summary>
        /// Gpu Is No Longer Reachable
        /// </summary>
        GpuIsLost = -18,

        /// <summary>
        /// Gpu Requires A Reset
        /// </summary>
        ResetRequired = -19,

        /// <summary>
        /// The Function That Was Requested Was Not Found (Bindings Only Error)
        /// </summary>
        FunctionNotFound = -20,

        /// <summary>
        /// The Connection To The Host Engine Is Not Valid Any Longer
        /// </summary>
        ConnectionNotValid = -21,

        /// <summary>
        /// This Gpu Is Not Supported By Dcgm
        /// </summary>
        GpuNotSupported = -22,

        /// <summary>
        /// The Gpus Of The Provided Group Are Not Compatible With Each Other For The Requested Operation
        /// </summary>
        GroupIncompatible = -23,

        /// <summary>
        /// Max Limit Reached For The Object
        /// </summary>
        MaxLimit = -24,

        /// <summary>
        /// Dcgm Library Could Not Be Found
        /// </summary>
        LibraryNotFound = -25,

        /// <summary>
        /// Duplicate Key Passed To A Function
        /// </summary>
        DuplicateKey = -26,

        /// <summary>
        /// Gpu Is Already A Part Of A Sync Boost Group
        /// </summary>
        GpuInSyncBoostGroup = -27,

        /// <summary>
        /// Gpu Is Not A Part Of A Sync Boost Group
        /// </summary>
        GpuNotInSyncBoostGroup = -28,

        /// <summary>
        /// This Operation Cannot Be Performed When The Host Engine Is Running As Non-Root
        /// </summary>
        RequiresRoot = -29,

        /// <summary>
        /// Dcgm Gpu Diagnostic Was Successfully Executed, But Reported An Error.
        /// </summary>
        NvvsError = -30,

        /// <summary>
        /// An Input Argument Is Not Large Enough
        /// </summary>
        InsufficientSize = -31,

        /// <summary>
        /// The Given Field Identifier Is Not Supported By The Api Being Called
        /// </summary>
        FieldUnsupportedByApi = -32,

        /// <summary>
        /// This Request Is Serviced By A Module Of Dcgm That Is Not Currently Loaded
        /// </summary>
        ModuleNotLoaded = -33,

        /// <summary>
        /// The Requested Operation Could Not Be Completed Because The Affected Resource Is In Use
        /// </summary>
        InUse = -34,

        /// <summary>
        /// This Group Is Empty And The Requested Operation Is Not Valid On An Empty Group
        /// </summary>
        GroupIsEmpty = -35,

        /// <summary>
        /// Profiling Is Not Supported For This Group Of Gpus Or Gpu.
        /// </summary>
        ProfilingNotSupported = -36,

        /// <summary>
        /// The Third-Party Profiling Module Returned An Unrecoverable Error.
        /// </summary>
        ProfilingLibraryError = -37,

        /// <summary>
        /// The Requested Profiling Metrics Cannot Be Collected In A Single Pass
        /// </summary>
        ProfilingMultiPass = -38,

        /// <summary>
        /// A Diag Instance Is Already Running, Cannot Run A New Diag Until The Current One Finishes.
        /// </summary>
        DiagAlreadyRunning = -39,

        /// <summary>
        /// The Dcgm Gpu Diagnostic Returned Json That Cannot Be Parsed
        /// </summary>
        DiagBadJson = -40,

        /// <summary>
        /// Error While Launching The Dcgm Gpu Diagnostic
        /// </summary>
        DiagBadLaunch = -41,

        /// <summary>
        /// There Is Too Much Variance While Training The Diagnostic
        /// </summary>
        DiagVariance = -42,

        /// <summary>
        /// A Field Value Met Or Exceeded The Error Threshold.
        /// </summary>
        DiagThresholdExceeded = -43,

        /// <summary>
        /// The Installed Driver Version Is Insufficient For This Api
        /// </summary>
        InsufficientDriverVersion = -44,
    };
}
