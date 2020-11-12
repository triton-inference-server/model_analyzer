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

from .mock_server import MockServerMethods
from unittest.mock import patch, Mock, MagicMock


class MockServerDockerMethods(MockServerMethods):
    """
    Mocks the docker module as used in 
    model_analyzer/triton/server/server_docker.py.
    Provides functions to check operation.
    """

    def __init__(self):
        docker_container_attrs = {'exec_run': MagicMock()}
        docker_client_attrs = {
            'containers.run': Mock(return_value=Mock(**docker_container_attrs))
        }
        docker_attrs = {
            'from_env': Mock(return_value=Mock(**docker_client_attrs)),
            'types.DeviceRequest': Mock(return_value=0)
        }
        self.patcher_docker = patch(
            'model_analyzer.triton.server.server_docker.docker',
            Mock(**docker_attrs))

        self.mock = self.patcher_docker.start()

    def stop(self):
        """
        Destroy the mocked classes and
        functions
        """
        self.patcher_docker.stop()

    def _assert_docker_initialized(self):
        """
        Asserts that docker.from_env
        was called to initialize
        docker client
        """

        self.mock.from_env.assert_called()

    def _assert_docker_exec_run_with_args(self, cmd, stream=True):
        """
        Asserts that a command cmd was run on the docker container
        with the given stream value
        """
        self.mock.from_env.return_value.containers.run.return_value.exec_run.assert_called_once_with(
            cmd, stream=True)

    def assert_server_process_start_called_with(self,
                                                cmd,
                                                local_model_path,
                                                model_repository_path,
                                                triton_image,
                                                http_port=8000,
                                                grpc_port=8001,
                                                metrics_port=8002):
        """
        Asserts that a triton container was created using the
        supplied arguments
        """

        self._assert_docker_initialized()

        mock_volumes = {
            local_model_path: {
                'bind': model_repository_path,
                'mode': 'rw'
            }
        }
        mock_ports = {http_port: 8000, grpc_port: 8001, metrics_port: 8002}
        self.mock.from_env.return_value.containers.run.assert_called_once_with(
            image=triton_image,
            device_requests=[0],
            volumes=mock_volumes,
            ports=mock_ports,
            publish_all_ports=True,
            tty=True,
            stdin_open=True,
            detach=True)

        self._assert_docker_exec_run_with_args(cmd=cmd, stream=True)

    def assert_server_process_terminate_called(self):
        """
        Asserts that stop was called on 
        the return value of containers.run
        """

        self.mock.from_env.return_value.containers.run.return_value.stop.assert_called(
        )
        self.mock.from_env.return_value.containers.run.return_value.remove.assert_called(
        )
