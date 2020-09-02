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
using System.Net;

namespace ModelAnalyzer.Client
{
    /// <summary>
    /// Exception Class to define exceptions to be thrown when an error is encountered
    /// when trying to load a model on an inference server.
    /// </summary>
    public class ModelLoadException : InvalidOperationException
    {
        internal ModelLoadException(string message, string modelName, IPAddress ipAddress, RequestStatusCode code)
            : base($"Error loading model \"{modelName}\" [{ipAddress}]: \"{message}\" ({code})")
        { }

        internal ModelLoadException(FormattableString message)
            : base(message?.ToString())
        { }
    }

    /// <summary>
    /// Exception Class to define exceptions to be thrown when an error is encountered
    /// when trying to unload a model from an inference server.
    /// </summary>
    public class ModelUnloadException : InvalidOperationException
    {
        internal ModelUnloadException(string message, string modelName, IPAddress ipAddress, RequestStatusCode code)
            : base($"Error unloading model \"{modelName}\" [[{ipAddress}]: \"{message}\" ({code})")
        { }

        internal ModelUnloadException(FormattableString message)
            : base(message?.ToString())
        { }
    }

    /// <summary>
    /// Exception Class to define exceptions to be thrown when an error is encountered
    /// when trying to get the status from an inference server.
    /// </summary>
    public class InferenceServerGetStatusException : InvalidOperationException
    {
        internal InferenceServerGetStatusException(string message, IPAddress ipAddress, RequestStatusCode code)
            : base($"Inference server [{ipAddress}] is unhealthy: \"{message}\" ({code})")
        { }

        internal InferenceServerGetStatusException(FormattableString message)
            : base(message?.ToString())
        { }
    }

    /// <summary>
    /// Exception Class to define exceptions to be thrown when an error is encountered
    /// when trying to parse an IP Address from an inference server.
    /// </summary>
    public class IPAddressParseException : FormatException
    {
        public IPAddressParseException(string message)
            : base(message)
        { }

        public IPAddressParseException(FormattableString message)
            : base(message?.ToString())
        { }
    }

    /// <summary>
    /// Exception Class to define exceptions to be thrown when an error is encountered
    /// when trying to acquire an inference server resource.
    /// </summary>
    public class InferenceServerAcquireException : InvalidOperationException
    {
        public InferenceServerAcquireException(string message)
            : base(message)
        { }

        public InferenceServerAcquireException(FormattableString message)
            : base(message?.ToString())
        { }
    }

    /// <summary>
    /// Exception Class to define exceptions to be thrown when an inference server pod is not ready within
    /// a timeout.
    /// </summary>
    public class InferenceServerPodNotReadyException : InvalidOperationException
    {
        public InferenceServerPodNotReadyException(string message)
            : base(message)
        { }

        public InferenceServerPodNotReadyException(FormattableString message)
            : base(message?.ToString())
        { }
    }
}
