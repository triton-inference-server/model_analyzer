# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from subprocess import Popen, PIPE, STDOUT, TimeoutExpired
import psutil
import logging

from .server import TritonServer
from model_analyzer.constants import SERVER_OUTPUT_TIMEOUT_SECS

logger = logging.getLogger(__name__)


class TritonServerLocal(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    tritonserver locally as as subprocess.
    """

    def __init__(self, path, config):
        """
        Parameters
        ----------
        path  : str
            The absolute path to the tritonserver executable
        config : TritonServerConfig
            the config object containing arguments for this server instance
        """

        self._tritonserver_process = None
        self._server_config = config
        self._server_path = path
        self._log = None

        assert self._server_config['model-repository'], \
            "Triton Server requires --model-repository argument to be set."

    def start(self):
        """
        Starts the tritonserver container locally
        """

        if self._server_path:
            # Create command list and run subprocess
            cmd = [self._server_path]
            cmd += self._server_config.to_cli_string().replace('=', ' ').split()
            self._tritonserver_process = Popen(cmd,
                                               start_new_session=True,
                                               stdout=PIPE,
                                               stderr=STDOUT,
                                               universal_newlines=True)

            logger.info('Triton Server started.')

    def stop(self):
        """
        Stops the running tritonserver
        """

        # Terminate process, capture output
        if self._tritonserver_process is not None:
            self._tritonserver_process.terminate()
            try:
                self._log, _ = self._tritonserver_process.communicate(
                    timeout=SERVER_OUTPUT_TIMEOUT_SECS)
            except TimeoutExpired:
                self._tritonserver_process.kill()
                self._log, _ = self._tritonserver_process.communicate()
            self._tritonserver_process = None
            logger.info('Triton Server stopped.')

    def logs(self):
        """
        Retrieves the Triton server's stdout
        as a str
        """

        return self._log

    def cpu_stats(self):
        """
        Returns the CPU memory usage and CPU available memory in MB
        """

        if self._tritonserver_process:
            server_process = psutil.Process(self._tritonserver_process.pid)
            process_memory_info = server_process.memory_full_info()
            system_memory_info = psutil.virtual_memory()

            # Divide by 1.0e6 to convert from bytes to MB
            return (process_memory_info.uss //
                    1.0e6), (system_memory_info.available // 1.0e6)
        else:
            return 0.0, 0.0
