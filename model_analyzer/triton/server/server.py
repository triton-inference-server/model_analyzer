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

from model_analyzer.output.file_writer import FileWriter

from abc import ABC, abstractmethod


class TritonServer(ABC):
    """
    Defines the interface for the objects created by
    TritonServerFactory
    """

    @abstractmethod
    def start(self):
        """
        Starts the tritonserver
        """

    @abstractmethod
    def stop(self):
        """
        Stops and cleans up after the server
        """

    @abstractmethod
    def logs(self):
        """
        Gets the server's stdout logs as a string
        """

    @abstractmethod
    def cpu_stats(self):
        """
        Returns the CPU memory usage and CPU available memory in MB
        """

    def update_config(self, params):
        """
        Update the server's arguments

        Parameters
        ----------
        params: dict
            keys are argument names and values are their values.
        """

        self._server_config.update_config(params)

    def write_server_logs(self, triton_output_path):
        """
        Checks if server logs have been
        requested, and writes them
        to the specified file

        Parameters
        ----------
        triton_output_path : str
            The full path to the triton output logs.
        """

        server_log_writer = FileWriter(filename=triton_output_path)
        server_log_writer.write(self.logs(), append=True)
