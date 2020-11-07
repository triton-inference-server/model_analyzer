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
                                                universal_newlines=True)

    def assert_server_process_terminate_called(self):
        """
        Asserts that terminate was called on
        the pipe (Popen object).
        """

        self.popen_mock.return_value.terminate.assert_called()
        self.popen_mock.return_value.communicate.assert_called()
