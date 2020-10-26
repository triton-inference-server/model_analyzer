# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from .server_docker import TritonServerDocker
from .server_local import TritonServerLocal


class TritonServerFactory:
    """
    A factory for creating TritonServer instances
    """
    @staticmethod
    def create_server_docker(model_path, version, config):
        """
        Parameters
        ----------
        model_path : str
            The absolute path to the local directory containing the models.
            In the case of locally running server, this may be the same as
            the model repository path
        version : str
            Current version of Triton Inference Server
        config : TritonServerConfig
            the config object containing arguments for this server instance
        
        Returns
        -------
        TritonServerDocker
        """
        return TritonServerDocker(model_path=model_path,
                                  version=version,
                                  config=config)

    @staticmethod
    def create_server_local(version, config):
        """
        Parameters
        ----------
        version : str
            Current version of Triton Inference Server
        config : TritonServerConfig
            the config object containing arguments for this server instance
        
        Returns
        -------
        TritonServerLocal
        """
        return TritonServerLocal(version=version, config=config)
