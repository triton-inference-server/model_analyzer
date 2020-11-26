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

from .mock_server import MockServerMethods
from unittest.mock import patch, Mock, MagicMock


class MockServerLocalMethods(MockServerMethods):
    """
    Mocks the subprocess functions used in 
    model_analyzer/triton/server/server_local.py.
    Provides functions to check operation.
    """

    def __init__(self):
        Popen_attrs = {'communicate.return_value': ('output', 'error')}
        self.patcher_popen = patch(
            'model_analyzer.triton.server.server_local.Popen',
            Mock(return_value=Mock(**Popen_attrs)))
        self.patcher_stdout = patch(
            'model_analyzer.triton.server.server_local.STDOUT', MagicMock())
        self.patcher_pipe = patch(
            'model_analyzer.triton.server.server_local.PIPE', MagicMock())
        self.popen_mock = self.patcher_popen.start()
        self.stdout_mock = self.patcher_stdout.start()
        self.pipe_mock = self.patcher_pipe.start()

    def stop(self):
        """
        Destroy the mocked classes and
        functions
        """

        self.patcher_popen.stop()
        self.patcher_stdout.stop()
        self.patcher_pipe.stop()

    def assert_server_process_start_called_with(self, cmd):
        """
        Asserts that Popen was called
        with the cmd provided.
        """

        self.popen_mock.assert_called_once_with(cmd,
                                                stdout=self.pipe_mock,
                                                stderr=self.stdout_mock,
                                                start_new_session=True,
                                                universal_newlines=True)

    def assert_server_process_terminate_called(self):
        """
        Asserts that terminate was called on
        the pipe (Popen object).
        """

        self.popen_mock.return_value.terminate.assert_called()
        self.popen_mock.return_value.communicate.assert_called()
