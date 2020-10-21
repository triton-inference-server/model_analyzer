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

import os
import docker

from .server import TritonServer

TRITONSERVER_IMAGE = 'nvcr.io/nvidia/tritonserver:'

LOCAL_HTTP_PORT = 8000
LOCAL_GRPC_PORT = 8001
LOCAL_METRICS_PORT = 8002


class TritonServerDocker(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    triton in a docker container.
    """

    def __init__(self, model_path, version, config):
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
        """
        self._version = version
        self._server_config = config
        self._model_path = model_path
        self._docker_client = docker.from_env()
        self._tritonserver_image = TRITONSERVER_IMAGE + version + '-py3'
        self._tritonserver_container = None

        assert self._server_config['model-repository'], \
            "Triton Server requires --model-repository argument to be set."

    def start(self):
        """
        Starts the tritonserver docker container using docker-py
        """
        # get devices using CUDA_VISIBLE_DEVICES
        CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')

        if CUDA_VISIBLE_DEVICES != '':
            device_ids = CUDA_VISIBLE_DEVICES.split(',')
            devices = [
                docker.types.DeviceRequest(
                    device_ids=device_ids,
                    capabilities=[['gpu']]
                )
            ]
        else:
            devices = None

        # Mount required directories
        volumes = {
            self._model_path: {
                'bind': self._server_config['model-repository'],
                'mode': 'rw'
            }
        }

        # Map ports, use config values but set to server defaults if not
        # specified
        server_http_port = self._server_config['http-port'] or 8000
        server_grpc_port = self._server_config['grpc-port'] or 8001
        server_metrics_port = self._server_config['metrics-port'] or 8002

        ports = {
            server_http_port: LOCAL_HTTP_PORT,
            server_grpc_port: LOCAL_GRPC_PORT,
            server_metrics_port: LOCAL_METRICS_PORT
        }

        # Run the docker container
        self._tritonserver_container = self._docker_client.containers.run(
            image=self._tritonserver_image,
            device_requests=devices,
            volumes=volumes,
            ports=ports,
            publish_all_ports=True,
            tty=True,
            stdin_open=True,
            detach=True)

        # Run the command in the container
        cmd = '/opt/tritonserver/bin/tritonserver ' + \
            self._server_config.to_cli_string()

        self._tritonserver_log = \
            self._tritonserver_container.exec_run(cmd, stream=True)

    def stop(self):
        """
        Stops the tritonserver docker container
        and cleans up docker client
        """
        if self._tritonserver_container is not None:
            self._tritonserver_container.stop()
            self._tritonserver_container.remove()

            self._tritonserver_container = None
            self._docker_client.close()
