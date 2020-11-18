# Copyright 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import docker

from .server import TritonServer

LOCAL_HTTP_PORT = 8000
LOCAL_GRPC_PORT = 8001
LOCAL_METRICS_PORT = 8002


class TritonServerDocker(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    triton in a docker container.
    """

    def __init__(self, model_path, image, config):
        """
        Parameters
        ----------
        model_path : str
            The absolute path to the local directory containing the models.
            In the case of locally running server, this may be the same as
            the model repository path
        image : str
            The tritonserver docker image to pull and run
        config : TritonServerConfig
            the config object containing arguments for this server instance
        """

        self._server_config = config
        self._model_path = model_path
        self._docker_client = docker.from_env()
        self._tritonserver_image = image
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
                docker.types.DeviceRequest(device_ids=device_ids,
                                           capabilities=[['gpu']])
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
        cmd = '/opt/tritonserver/bin/tritonserver ' + self._server_config.to_cli_string(
        )

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
