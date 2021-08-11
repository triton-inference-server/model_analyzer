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

from .server import TritonServer
from model_analyzer.constants import SERVER_OUTPUT_TIMEOUT_SECS
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from subprocess import Popen, DEVNULL, STDOUT, TimeoutExpired
import psutil
import logging
import os

logger = logging.getLogger(__name__)


class TritonServerLocal(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    tritonserver locally as as subprocess.
    """
    def __init__(self, path, config, gpus, log_path):
        """
        Parameters
        ----------
        path  : str
            The absolute path to the tritonserver executable
        config : TritonServerConfig
            the config object containing arguments for this server instance
        gpus: list of str
            List of GPU UUIDs to be made visible to Triton
        log_path: str
            Absolute path to the triton log file
        """

        self._tritonserver_process = None
        self._server_config = config
        self._server_path = path
        self._gpus = gpus
        self._log_path = log_path

        assert self._server_config['model-repository'], \
            "Triton Server requires --model-repository argument to be set."

    def start(self, env=None):
        """
        Starts the tritonserver container locally
        """

        if self._server_path:
            # Create command list and run subprocess
            cmd = [self._server_path]
            cmd += self._server_config.to_cli_string().replace('=',
                                                               ' ').split()
            # Set environment, update with user config env
            triton_env = os.environ.copy()

            if env:
                # Filter env variables that use env lookups
                for variable, value in env.items():
                    if value.find('$') == -1:
                        triton_env[variable] = value
                    else:
                        # Collect the ones that need lookups to give to the shell
                        triton_env[variable] = os.path.expandvars(value)

            # List GPUs to be used by tritonserver
            triton_env['CUDA_VISIBLE_DEVICES'] = ','.join(
                [uuid for uuid in self._gpus])

            if self._log_path:
                try:
                    self._log_file = open(self._log_path, 'a+')
                except OSError as e:
                    raise TritonModelAnalyzerException(e)
            else:
                self._log_file = DEVNULL

            # Construct Popen command
            self._tritonserver_process = Popen(cmd,
                                               stdout=self._log_file,
                                               stderr=STDOUT,
                                               start_new_session=True,
                                               universal_newlines=True,
                                               env=triton_env)

            logger.info('Triton Server started.')

    def stop(self):
        """
        Stops the running tritonserver
        """

        # Terminate process, capture output
        if self._tritonserver_process is not None:
            self._tritonserver_process.terminate()
            try:
                self._tritonserver_process.communicate(
                    timeout=SERVER_OUTPUT_TIMEOUT_SECS)
            except TimeoutExpired:
                self._tritonserver_process.kill()
                self._tritonserver_process.communicate()
            self._tritonserver_process = None
            if self._log_path:
                self._log_file.close()
            logger.info('Triton Server stopped.')

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
