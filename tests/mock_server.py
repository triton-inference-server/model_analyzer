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

from abc import ABC, abstractmethod
from unittest.mock import patch, Mock, MagicMock


class MockServerMethods(ABC):
    """
    Interface for a mock server declaring
    the methods it must provide.
    """

    @abstractmethod
    def stop(self):
        """
        Destroy the mocked classes and
        functions
        """

    @abstractmethod
    def assert_server_process_start_called_with(self, **args):
        """
        Asserts that the tritonserver process was started with
        the supplied arguments
        """

    @abstractmethod
    def assert_server_process_terminate_called(self):
        """
        Assert that the server process was stopped
        """
