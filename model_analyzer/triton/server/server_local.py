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
import logging

from .server import TritonServer

SERVER_OUTPUT_TIMEOUT_SECS = 5
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

        assert self._server_config['model-repository'], \
            "Triton Server requires --model-repository argument to be set."

    def start(self):
        """
        Starts the tritonserver container locally
        """

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
            logger.info('Triton Server stopped.')
            self._tritonserver_process.terminate()
            try:
                output, _ = self._tritonserver_process.communicate(
                    timeout=SERVER_OUTPUT_TIMEOUT_SECS)
            except TimeoutExpired:
                self._tritonserver_process.kill()
                output, _ = self._tritonserver_process.communicate()
            self._tritonserver_process = None
