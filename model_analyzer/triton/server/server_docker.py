# Copyright (c) 2020,21 NVIDIA CORPORATION. All rights reserved.
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

import docker
import logging
from multiprocessing.pool import ThreadPool

from .server import TritonServer
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

LOCAL_HTTP_PORT = 8000
LOCAL_GRPC_PORT = 8001
LOCAL_METRICS_PORT = 8002

logger = logging.getLogger(__name__)


class TritonServerDocker(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    triton in a docker container.
    """

    def __init__(self, image, config, gpus, log_path):
        """
        Parameters
        ----------
        image : str
            The tritonserver docker image to pull and run
        config : TritonServerConfig
            the config object containing arguments for this server instance
        gpus : list of str
            List of GPU UUIDs to be mounted and used in the container
        log_path: str
            Absolute path to the triton log file
        """

        self._server_config = config
        self._docker_client = docker.from_env()
        self._tritonserver_image = image
        self._tritonserver_container = None
        self._log_path = log_path
        self._gpus = gpus

        assert self._server_config['model-repository'], \
            "Triton Server requires --model-repository argument to be set."

    def start(self):
        """
        Starts the tritonserver docker container using docker-py
        """

        # List GPUs to be mounted and used inside docker container
        devices = []
        if len(self._gpus):
            devices = [
                docker.types.DeviceRequest(device_ids=self._gpus,
                                           capabilities=[['gpu']])
            ]
        environment = {
            'CUDA_VISIBLE_DEVICES': ','.join([uuid for uuid in self._gpus])
        }

        # Mount required directories
        volumes = {
            self._server_config['model-repository']: {
                'bind': self._server_config['model-repository'],
                'mode': 'ro'
            }
        }

        # Map ports, use config values but set to server defaults if not
        # specified
        server_http_port = self._server_config['http-port'] or 8000
        server_grpc_port = self._server_config['grpc-port'] or 8001
        server_metrics_port = self._server_config['metrics-port'] or 8002

        ports = {
            server_http_port: server_http_port,
            server_grpc_port: server_grpc_port,
            server_metrics_port: server_metrics_port
        }

        try:
            # Run the docker container and run the command in the container
            self._tritonserver_container = self._docker_client.containers.run(
                command='tritonserver ' + self._server_config.to_cli_string(),
                name='tritonserver',
                image=self._tritonserver_image,
                device_requests=devices,
                environment=environment,
                volumes=volumes,
                ports=ports,
                publish_all_ports=True,
                tty=False,
                stdin_open=False,
                detach=True)
        except docker.errors.APIError as error:
            if error.explanation.find('port is already allocated') != -1:
                raise TritonModelAnalyzerException(
                    "One of the following port(s) are already allocated: "
                    f"{server_http_port}, {server_grpc_port}, "
                    f"{server_metrics_port}.\n"
                    "Change the Triton server ports using"
                    " --triton-http-endpoint, --triton-grpc-endpoint,"
                    " and --triton-metrics-endpoint flags.")
            else:
                raise error

        if self._log_path:
            try:
                self._log_file = open(self._log_path, 'a+')
                self._log_pool = ThreadPool(processes=1)
                self._log_pool.apply_async(self._logging_worker)
            except OSError as e:
                raise TritonModelAnalyzerException(e)

    def _logging_worker(self):
        """
        streams logs to
        log file
        """

        for chunk in self._tritonserver_container.logs(stream=True):
            self._log_file.write(chunk.decode('utf-8'))

    def stop(self):
        """
        Stops the tritonserver docker container
        and cleans up docker client
        """

        logger.info('Stopping triton server.')

        if self._tritonserver_container is not None:
            if self._log_path:
                if self._log_pool:
                    self._log_pool.terminate()
                    self._log_pool.close()
                if self._log_file:
                    self._log_file.close()
            self._tritonserver_container.stop()
            self._tritonserver_container.remove(force=True)
            self._tritonserver_container = None
            self._docker_client.close()

    def cpu_stats(self):
        """
        Returns the CPU memory usage and CPU available memory in MB
        """

        cmd = 'bash -c "pmap -x $(pgrep tritonserver) | tail -n1 | awk \'{print $4}\'"'
        _, used_mem_bytes = self._tritonserver_container.exec_run(cmd=cmd,
                                                                  stream=False)
        cmd = 'bash -c "free | awk \'{if(NR==2)print $7}\'"'
        _, available_mem_bytes = self._tritonserver_container.exec_run(
            cmd=cmd, stream=False)

        # Divide by 1.0e6 to convert from kilobytes to MB
        return float(used_mem_bytes.decode("utf-8")) // 1.0e3, float(
            available_mem_bytes.decode("utf-8")) // 1.0e3
