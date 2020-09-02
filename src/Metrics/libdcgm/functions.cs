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
    static unsafe partial class libdcgm
    {
        /// <summary>
        /// This method is used to connect to a stand-alone host engine process. Remote host engines are started by running the nv-hostengine command.
        /// <para/>
        /// NOTE: dcgmConnect_v2 provides additional connection options.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="ipAddress">
        /// Valid IP address for the remote host engine to connect to.
        /// <para/>
        /// If ipAddress is specified as x.x.x.x it will attempt to connect to the default port specified by DCGM_HE_PORT_NUMBER.
        /// <para/>
        /// If ipAddress is specified as x.x.x.x:yyyy it will attempt to connect to theport specified by yyyy.
        /// </param>
        /// <param name="dcgmHandle">
        /// DCGM Handle of the remote host engine.
        /// </param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmConnect))]
        public static extern dcgm_return dcgmConnect(
            [In] byte* ipAddress,
            [Out] IntPtr* dcgmHandle);


        /// <summary>
        /// This method is used to disconnect from a stand-alone host engine process.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">DCGM Handle that came form <see cref="dcgmConnect"/>.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmDisconnect))]
        public static extern dcgm_return dcgmDisconnect(
            [In] IntPtr dcgmHandle);


        /// <summary>
        /// Get a pointer to the metadata for a field by its field Identifier. See DCGM_FI_? for a list of field IDs.
        /// <para/>
        /// Returns 0 on failure and >0 if Pointer to field metadata structure is found.
        /// </summary>
        /// <param name="fieldId">One of the field IDs (DCGM_FI_?)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(DcgmFieldGetById))]
        public static extern dcgm_field_meta* DcgmFieldGetById(
            [In] ushort fieldId);


        /// <summary>
        /// Used to create a group of fields and return the handle in dcgmFieldGroupId
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">DCGM Handle.</param>
        /// <param name="numFieldIds">Number of field IDs that are being provided in fieldIds[]. Must be between 1 and DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP.</param>
        /// <param name="fieldIds">Field IDs to be added to the newly-created field group.</param>
        /// <param name="fieldGroupName">Unique name for this group of fields. This must not be the same as any existing field groups.</param>
        /// <param name="dcgmFieldGroupId">Handle to the newly-created field group.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmFieldGroupCreate))]
        public static extern dcgm_return dcgmFieldGroupCreate(
            [In] IntPtr dcgmHandle,
            [In] int numFieldIds,
            [In] ushort* fieldIds,
            [In] byte* fieldGroupName,
            [Out] IntPtr* dcgmFieldGroupId);


        /// <summary>
        /// Used to remove a field group that was created with <see cref="dcgm.dcgmFieldGroupCreate"/>
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">DCGM handle</param>
        /// <param name="dcgmFieldGroupId">Field group to remove</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmFieldGroupDestroy))]
        public static extern dcgm_return dcgmFieldGroupDestroy(
            [In] IntPtr dcgmHandle,
            [In] IntPtr dcgmFieldGroupId);


        /// <summary>
        /// This method is used to get identifiers corresponding to all the DCGM-supported devices on the system.
        /// <para/>
        /// The identifier represents DCGM GPU Identifier corresponding to each GPU on the system and is immutable during the lifespan of the engine.
        /// <para/>
        /// The list should be queried again if the engine is restarted.
        /// <para/>
        /// The GPUs returned from this function ONLY includes gpuIds of GPUs that are supported by DCGM.
        /// <para/>
        /// To get gpuIds of all GPUs in the system, use <seealso cref="dcgmGetAllDevices()"/>.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">DCGM Handle.</param>
        /// <param name="gpuIdList">Array reference to fill GPU Ids present on the system.</param>
        /// <param name="count">Number of GPUs returned in <paramref name="gpuIdList"/>.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmGetAllSupportedDevices))]
        public static extern dcgm_return dcgmGetAllSupportedDevices(
            [In] IntPtr dcgmHandle,
            [Out] uint* gpuIdList,
            [Out] int* count);


        /// <summary>
        /// Request latest cached field value for a GPU
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>

        /// <param name="dcgmHandle">DCGM Handle</param>
        /// <param name="gpuId">Gpu Identifier representing the GPU for which the fields are being requested.</param>
        /// <param name="fields">Field IDs to return data for. See the definitions in dcgm_fields.h that start with DCGM_FI_.</param>
        /// <param name="count">Number of field IDs in fields[] array.</param>
        /// <param name="values">Latest field values for the fields in fields[].</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmGetLatestValuesForFields))]
        public static extern dcgm_return dcgmGetLatestValuesForFields(
            [In] IntPtr dcgmHandle,
            [In] uint gpuId,
            [In] ushort* fields,
            [In] uint count,
            [Out] dcgm_field_value_v1* values);


        /// <summary>
        /// Used to add specified GPU Id to the group represented by <paramref name="groupId"/>.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">DCGM Handle.</param>
        /// <param name="groupId">Group Id to which device should be added.</param>
        /// <param name="gpuId">DCGM GPU Id.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmGroupAddDevice))]
        public static extern dcgm_return dcgmGroupAddDevice(
            [In] IntPtr dcgmHandle,
            [In] IntPtr dcgmGroupId,
            [In] uint gpuId);


        /// <summary>
        /// Used to create a entity group handle which can store one or more entity Ids as an opaque handle returned in <paramref name="dcgmGroupId"/>.
        /// <para/>
        /// Instead of executing an operation separately for each entity, the DCGM group enables the user to execute same operation on all the entities present in the group as a single API call.
        /// <para/>
        /// To create the group with all the entities present on the system, the <paramref name="type"/> field should be specified as <seealso cref="dcgm_group_type.Default"/> or <seealso cref="dcgm_group_type.NvSwitches"/>.
        /// <para/>
        /// To create an empty group, the <paramref name="type"/> field should be specified as <seealso cref="dcgm_group_type.Empty"/>.
        /// <para/>
        /// The empty group can be updated with the desired set of entities using the APIs <seealso cref="dcgmGroupAddDevice"/>, <seealso cref="dcgmGroupAddEntity"/>,  <seealso cref="dcgmGroupRemoveDevice"/>, and <seealso cref="dcgmGroupRemoveEntity"/>.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">DCGM Handle.</param>
        /// <param name="type">Type of Entity Group to be formed.</param>
        /// <param name="groupName">Desired name of the GPU group specified as NULL terminated C string.</param>
        /// <param name="dcgmGroupId">Reference to group ID.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmGroupCreate))]
        public static extern dcgm_return dcgmGroupCreate(
            [In] IntPtr dcgmHandle,
            [In] dcgm_group_type type,
            [In] byte* groupName,
            [Out] IntPtr* dcgmGroupId);


        /// <summary>
        /// Used to destroy a group represented by a groupId.
        /// <para/>
        /// Since DCGM group is a logical grouping of entities, the properties applied on the group stay intact
        /// for the individual entities even after the group is destroyed.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">DCGM Handle.</param>
        /// <param name="groupId">Group ID.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmGroupDestroy))]
        public static extern dcgm_return dcgmGroupDestroy(
            [In] IntPtr dcgmHandle,
            [In] IntPtr pDcgmGrpId);


        /// <summary>
        /// Used to get information corresponding to the group represented by groupId. The information
        /// returned in pDcgmGroupInfo consists of group name, and the list of entities present in the
        /// group.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">DCGM Handle</param>
        /// <param name="groupId">Group ID for which information to be fetched</param>
        /// <param name="pDcgmGroupInfo">Group Information</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmGroupGetInfo))]
        public static extern dcgm_return dcgmGroupGetInfo(
            [In] IntPtr dcgmHandle,
            [In] IntPtr groupId,
            [Out] dcgm_group_info_v2* pDcgmGroupInfo);


        /// <summary>
        /// Used to get information corresponding to the field group represented by field group identifier.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">DCGM Handle</param>
        /// <param name="pFieldGroupInfo">
        /// Field Group Information. Version should be set before this call. pFieldGroupInfo->fieldGroupId should contain the fieldGroupId you are interested in querying information for.
        /// </param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmFieldGroupGetInfo))]
        public static extern dcgm_return dcgmFieldGroupGetInfo(
            [In] IntPtr dcgmHandle,
            [Out] dcgm_field_group_info_v1* pFieldGroupInfo);


        /// <summary>
        /// Enable the DCGM health check system for the given systems defined in <see cref="dcgm_health_systems_watch"/>
        /// group.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        ///
        /// <param name="dcgmHandle">DCGM Handle</param>
        /// <param name="groupId">Group ID representing collection of one or more entities. Look at
        ///                       <see cref="dcgm.dcgmGroupCreate"/> for details on creating the group.</param>
        /// <param name="systems">An enum representing systems that should be enabled for health
        ///                       checks logically OR together. Refer to <see cref="dcgm_health_systems_watch"/>
        ///                       for details.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmHealthSet))]
        public static extern dcgm_return dcgmHealthSet(
            [In] IntPtr dcgmHandle,
            [In] IntPtr groupId,
            [In] dcgm_health_systems_watch systems);


        /// <summary>
        /// Used to initialize DCGM within this process. This must be called before <seealso cref="dcgmStartEmbedded()"/> or <seealso cref="dcgmConnect()"/>.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmInit))]
        public static extern dcgm_return dcgmInit();


        /// <summary>
        /// This method is used to shut down DCGM. Any embedded host engines or remote connections will automatically
        /// be shut down as well.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmShutdown))]
        public static extern dcgm_return dcgmShutdown();


        /// <summary>
        /// Start an embedded host engine agent within this process.
        /// <para/>
        /// The agent is loaded as a shared library.
        /// <para/>
        /// This mode is provided to avoid any extra jitter associated with an additional autonomous agent needs to be managed.
        /// <para/>
        /// In this mode, the user has to periodically call APIs such as <seealso cref="dcgmPolicyTrigger"/> and <seealso cref="dcgmUpdateAllFields"/> which tells DCGM to wake up and perform data collection and operations needed for policy management.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="opMode">
        /// Collect data automatically or manually when asked by the user.
        /// </param>
        /// <param name="dcgmHandle">DCGM Handle to use for API calls.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmStartEmbedded))]
        public static extern dcgm_return dcgmStartEmbedded(
            [In] dcgm_operation_mode opMode,
            [Out] IntPtr* dcgmHandle);


        /// <summary>
        /// Stop the embedded host engine within this process that was started with dcgmStartEmbedded
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">
        /// DCGM Handle of the embedded host engine that came from <seealso cref="dcgmStartEmbedded"/>.
        /// </param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmStopEmbedded))]
        public static extern dcgm_return dcgmStopEmbedded(
            [In] IntPtr dcgmHandle);


        /// <summary>
        /// Request that DCGM stop recording updates for a given field collection.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">DCGM Handle</param>
        /// <param name="groupId">
        /// Group ID representing collection of one or more entities. Look at
        /// <see cref="dcgm.dcgmGroupCreate"/> for details on creating the group.
        /// </param>
        /// <param name="fieldGroupId">Fields to watch.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmUnwatchFields))]
        public static extern dcgm_return dcgmUnwatchFields(
            [In] IntPtr dcgmHandle,
            [In] IntPtr groupId,
            [In] IntPtr fieldGroupId);


        /// <summary>
        /// This method is used to tell the DCGM module to update all the fields being watched.
        /// <para/>
        /// Note: If the if the operation mode was set to manual mode (DCGM_OPERATION_MODE_MANUAL) during
        /// initialization <see cref="dcgm.dcgmInit"/>, this method must be caused periodically to allow field value watches
        /// the opportunity to gather samples.
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        ///
        /// <param name="dcgmHandle">DCGM Handle</param>
        /// <param name="waitForUpdate">Whether or not to wait for the update loop to complete before returning to the caller 1=wait. 0=do not wait.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmUpdateAllFields))]
        public static extern dcgm_return dcgmUpdateAllFields(
            [In] IntPtr dcgmHandle,
            [In] int waitForUpdate);


        /// <summary>
        /// Request that DCGM start recording updates for a given field collection.
        /// <para/>
        /// Note that the first update of the field will not occur until the next field update cycle.
        /// To force a field update cycle, call dcgmUpdateAllFields(1).
        /// <para/>
        /// Returns <see cref="dcgm_return.Ok"/> if successful; otherwise another value specifying the error.
        /// </summary>
        /// <param name="dcgmHandle">DCGM Handle</param>
        /// <param name="groupId">Group ID representing collection of one or more entities. Look at
        ///                       <see cref="dcgm.dcgmGroupCreate"/> for details on creating the group.</param>
        /// <param name="fieldGroupId">Fields to watch.</param>
        /// <param name="updateFreq">How often to update this field in usec</param>
        /// <param name="maxKeepAge">How long to keep data for this field in seconds</param>
        /// <param name="maxKeepSamples">Maximum number of samples to keep. 0=no limit</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = nameof(dcgmWatchFields))]
        public static extern dcgm_return dcgmWatchFields(
            [In] IntPtr dcgmHandle,
            [In] IntPtr groupId,
            [In] IntPtr fieldGroupId,
            [In] ulong updateFreq,
            [In] double maxKeepAge,
            [In] int maxKeepSamples);
    }
}
