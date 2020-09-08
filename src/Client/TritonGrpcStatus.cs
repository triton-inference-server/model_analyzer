/*****************************************************************************
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

using System;
using System.Collections.Generic;

using static System.StringComparer;

namespace Triton.MemoryAnalyzer.Client
{
    /// <summary>
    /// Interface to an object which defines the inference server status returned by making a
    /// grpc GetStatus call to the inference server.
    /// </summary>
    public interface ITritonStatus
    {
        /// <summary>
        /// Dictionary holding the status of all models on the inference server.
        /// </summary>
        IDictionary<string, IDictionary<string, ModelReadyState>> ModelStatus { get; }

        /// <summary>
        /// Ready state of the inference server.
        /// </summary>
        ServerReadyState ReadyState { get; }
    }

    public class TritonGrpcStatus : ITritonStatus
    {
        private readonly ServerStatus _status;
        private readonly Dictionary<string, IDictionary<string, ModelReadyState>> _modelStatus;

        public TritonGrpcStatus(ServerStatus status)
        {
            if (status is null)
            {
                throw new ArgumentNullException(nameof(status));
            }

            _status = status;
            _modelStatus = new Dictionary<string, IDictionary<string, ModelReadyState>>();

            foreach (var model in status.ModelStatus)
            {
                var versionStatus = new Dictionary<string, ModelReadyState>(Ordinal);

                foreach (var kvp in model.Value.VersionStatus)
                {
                    versionStatus[kvp.Key.ToString("N0")] = kvp.Value.ReadyState;
                }

                _modelStatus[model.Key] = versionStatus;
            }
        }

        public IDictionary<string, IDictionary<string, ModelReadyState>> ModelStatus
        {
            get { return _modelStatus; }
        }

        public ServerReadyState ReadyState
        {
            get { return _status.ReadyState; }
        }
    }
}
