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
    internal static class Utils
    {
        internal static uint dcgmConfigVersion()
        {
            return makeDcgmVersion(Marshal.SizeOf(new dcgm_config_v1 { }), 1);
        }

        internal static uint dcgmGroupInfoVersion()
        {
            return makeDcgmVersion(Marshal.SizeOf(new dcgm_group_info_v2 { }), 2);
        }

        internal static uint dcgmHealthResponseVersion1()
        {
            return makeDcgmVersion(Marshal.SizeOf(new dcgm_health_response_v1 { }), 1);
        }

        internal static uint dcgmFieldGroupInfoVersion1()
        {
            return makeDcgmVersion(Marshal.SizeOf(new dcgm_field_group_info_v1 { }), 1);
        }

        internal static string errorString(dcgm_return result)
        {
            switch (result)
            {
                case dcgm_return.Ok:
                    return "Success";
                case dcgm_return.BadParam:
                    return "Bad parameter passed to function";
                case dcgm_return.GenericError:
                    return "Generic unspecified error";
                case dcgm_return.Memory:
                    return "Out of memory error";
                case dcgm_return.NotConfigured:
                    return "Setting not configured";
                case dcgm_return.NotSupported:
                    return "Feature not supported";
                case dcgm_return.InitError:
                    return "DCGM initialization error";
                case dcgm_return.NvmlError:
                    return "NVML error";
                case dcgm_return.Pending:
                    return "Object is in a pending state";
                case dcgm_return.Uninitialized:
                    return "Object is in an undefined state";
                case dcgm_return.Timeout:
                    return "Timeout";
                case dcgm_return.VersionMismatch:
                    return "API version mismatch";
                case dcgm_return.UnknownField:
                    return "Unknown field identifier";
                case dcgm_return.NoData:
                    return "No data is available";
                case dcgm_return.StaleData:
                    return "Only stale data is available";
                case dcgm_return.NotWatched:
                    return "Field is not being watched";
                case dcgm_return.NoPermission:
                    return "No permission";
                case dcgm_return.GpuIsLost:
                    return "GPU is lost";
                case dcgm_return.ResetRequired:
                    return "GPU requires reset";
                case dcgm_return.ConnectionNotValid:
                    return "Host engine connection invalid/disconnected";
                case dcgm_return.GpuNotSupported:
                    return "This GPU is not supported by DCGM";
                case dcgm_return.GroupIncompatible:
                    return "The GPUs of this group are incompatible with each other for the requested operation";
                case dcgm_return.MaxLimit:
                    return "Max limit reached for the object";
                case dcgm_return.LibraryNotFound:
                    return "DCGM library could not be found";
                case dcgm_return.DuplicateKey:
                    return "Duplicate Key passed to function";
                case dcgm_return.GpuInSyncBoostGroup:
                    return "GPU is a part of a Sync Boost Group";
                case dcgm_return.GpuNotInSyncBoostGroup:
                    return "GPU is not a part of Sync Boost Group";
                case dcgm_return.RequiresRoot:
                    return "Host engine is running as non-root";
                case dcgm_return.NvvsError:
                    return "DCGM GPU Diagnostic returned an error";
                case dcgm_return.InsufficientSize:
                    return "An input argument is not large enough";
                case dcgm_return.FieldUnsupportedByApi:
                    return "The given field ID is not supported by the API being called";
                case dcgm_return.ModuleNotLoaded:
                    return "This request is serviced by a module of DCGM that is not currently loaded";
                case dcgm_return.InUse:
                    return "The requested operation could not be completed because the affected resource is in use";
                case dcgm_return.GroupIsEmpty:
                    return "The specified group is empty, and this operation is incompatible with an empty group";
                case dcgm_return.ProfilingNotSupported:
                    return "Profiling is not supported for this group of GPUs or GPU";
                case dcgm_return.ProfilingLibraryError:
                    return "The third-party Profiling module returned an unrecoverable error";
                case dcgm_return.ProfilingMultiPass:
                    return "The requested profiling metrics cannot be collected in a single pass";
                case dcgm_return.DiagAlreadyRunning:
                    return "A diag instance is already running, cannot run a new diag until the current one finishes";
                case dcgm_return.DiagBadJson:
                    return "The GPU Diagnostic returned Json that cannot be parsed.";
                case dcgm_return.DiagBadLaunch:
                    return "Error while launching the GPU Diagnostic.";
                case dcgm_return.DiagVariance:
                    return "The results of training DCGM GPU Diagnostic cannot be trusted because they vary too much from run to run";
                case dcgm_return.DiagThresholdExceeded:
                    return "A field value met or exceeded the error threshold.";
                case dcgm_return.InsufficientDriverVersion:
                    return "The installed driver version is insufficient for this API";
                default:
                    // Wrong error codes should be handled by the caller
                    return "";
            }
        }

        internal static uint makeDcgmVersion(int size, int ver)
        {
            return (uint)size | (uint)(ver) << 24;
        }
    }
}
