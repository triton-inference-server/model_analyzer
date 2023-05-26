# Copyright (c) 2020-2022 NVIDIA CORPORATION. All rights reserved.
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
from model_analyzer.constants import LOGGER_NAME, SERVER_OUTPUT_TIMEOUT_SECS
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from subprocess import Popen, DEVNULL, STDOUT, TimeoutExpired
import psutil
import logging
import tempfile
import os
from io import TextIOWrapper

logger = logging.getLogger(LOGGER_NAME)


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
        self._log_file = DEVNULL
        self._is_first_time_starting_server = True

        assert self._server_config['model-repository'], \
            "Triton Server requires --model-repository argument to be set."

    def start(self, env=None):
        """
        Starts the tritonserver container locally
        """

        if self._server_path:
            # Create command list and run subprocess
            cmd = [self._server_path]
            cmd += self._server_config.to_args_list()

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
                [gpu.device_uuid() for gpu in self._gpus])

            if self._log_path:
                try:
                    if self._is_first_time_starting_server:
                        if os.path.exists(self._log_path):
                            os.remove(self._log_path)
                    self._log_file = open(self._log_path, 'a+')
                except OSError as e:
                    raise TritonModelAnalyzerException(e)
            else:
                self._log_file = tempfile.NamedTemporaryFile()

            self._is_first_time_starting_server = False

            # Construct Popen command
            try:
                self._tritonserver_process = Popen(cmd,
                                                   stdout=self._log_file,
                                                   stderr=STDOUT,
                                                   start_new_session=True,
                                                   universal_newlines=True,
                                                   env=triton_env)

                logger.debug('Triton Server started.')
            except Exception as e:
                raise TritonModelAnalyzerException(e)

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
            logger.debug('Stopped Triton Server.')

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

    def log_file(self) -> TextIOWrapper:
        return self._log_file
