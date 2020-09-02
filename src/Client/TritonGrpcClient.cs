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

using Grpc.Core;
using System;
using System.Collections.Generic;
using System.Net;
using System.Threading.Tasks;

namespace ModelAnalyzer.Client
{
    /// <summary>
    /// Interface to an object which abstracts the client calls to the Inference Server.
    /// </summary>
    public interface ITritonClient
    {
        /// <summary>
        /// Loads a list of models on an Inference Server.
        /// </summary>
        /// <param name="models">List of models to be loaded.</param>
        /// <param name="ipAddress">Ip Address of the inference server.</param>
        /// <param name="port">Grpc Port of the inference server.</param>
        void LoadModels(IList<string> models, IPAddress ipAddress, int port);

        /// <summary>
        /// Unloads a list of models from an Inference Server.
        /// </summary>
        /// <param name="models">List of models to be unloaded.</param>
        /// <param name="ipAddress">Ip Address of the inference server.</param>
        /// <param name="port">Grpc Port of the inference server.</param>
        void UnloadModels(IList<string> models, IPAddress ipAddress, int port);

        /// <summary>
        /// Gets the status of an Inference Server.
        /// </summary>
        /// <param name="ipAddress">Ip Address of the inference server.</param>
        /// <param name="port">Grpc Port of the inference server.</param>
        ITritonStatus GetStatus(IPAddress ipAddress, int port);
    }

    public class TritonGrpcClient : ITritonClient
    {
        public TritonGrpcClient()
        { }

        public ITritonStatus GetStatus(IPAddress ipAddress, int port)
        {
            if (ipAddress is null)
            {
                throw new ArgumentNullException(nameof(ipAddress));
            }

            var target = $"{ipAddress}:{port}";
            var channel = new Channel(target, ChannelCredentials.Insecure);

            try
            {
                var grpcClient = new GRPCService.GRPCServiceClient(channel);

                var response = grpcClient.Status(new StatusRequest());

                if (response is null)
                    throw new InvalidOperationException("Grpc client failed to respond; the response was null.");
                
                if (response.RequestStatus.Code != RequestStatusCode.Success)
                    throw new InferenceServerGetStatusException(response.RequestStatus.Msg, ipAddress, response.RequestStatus.Code);

                return new TritonGrpcStatus(response.ServerStatus);
            }
            finally
            {
                var task = Task.Run(async () =>
                {
                    await channel.ShutdownAsync();
                });

                task.Wait();
            }
        }
        public void LoadModels(IList<string> models, IPAddress ipAddress, int port)
        {
            if (models is null)
                throw new ArgumentNullException(nameof(models));
            if (ipAddress is null)
                throw new ArgumentNullException(nameof(ipAddress));
            if (models.Count < 1)
                return;

            var target = $"{ipAddress}:{port}";
            var channel = new Channel(target, ChannelCredentials.Insecure);

            try
            {
                var grpcClient = new GRPCService.GRPCServiceClient(channel);

                foreach (var modelName in models)
                {
                    var request = new ModelControlRequest()
                    {
                        ModelName = modelName,
                        Type = ModelControlRequest.Types.Type.Load
                    };

                    var response = grpcClient.ModelControl(request);

                    if (response.RequestStatus.Code != RequestStatusCode.Success && response.RequestStatus.Code != RequestStatusCode.AlreadyExists)
                        throw new ModelLoadException(response.RequestStatus.Msg, modelName, ipAddress, response.RequestStatus.Code);
                }
            }
            finally
            {
                var task = Task.Run(async () =>
                {
                    await channel.ShutdownAsync();
                });

                task.Wait();
            }
        }

        public void UnloadModels(IList<string> models, IPAddress ipAddress, int port)
        {
            if (models is null)
                throw new ArgumentNullException(nameof(models));
            if (ipAddress is null)
                throw new ArgumentNullException(nameof(ipAddress));
            if (models.Count < 1)
                return;

            var target = $"{ipAddress}:{port}";
            var channel = new Channel(target, ChannelCredentials.Insecure);

            try
            {
                var grpcClient = new GRPCService.GRPCServiceClient(channel);

                foreach (var modelName in models)
                {
                    var request = new ModelControlRequest()
                    {
                        ModelName = modelName,
                        Type = ModelControlRequest.Types.Type.Unload
                    };

                    var response = grpcClient.ModelControl(request);

                    if (response.RequestStatus.Code != RequestStatusCode.Success && response.RequestStatus.Code != RequestStatusCode.AlreadyExists)
                        throw new ModelUnloadException(response.RequestStatus.Msg, modelName, ipAddress, response.RequestStatus.Code);
                }
            }
            finally
            {
                var task = Task.Run(async () =>
                {
                    await channel.ShutdownAsync();
                });

                task.Wait();
            }
        }
    }
}
