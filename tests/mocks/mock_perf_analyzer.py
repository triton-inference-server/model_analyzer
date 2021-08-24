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

from .mock_base import MockBase
from unittest.mock import patch, Mock, MagicMock


class MockCalledProcessError(Exception):
    """
    A mock of subprocess.CalledProcessError
    """

    def __init__(self):
        self.returncode = 1
        self.cmd = ["dummy command"]
        self.output = "mock output"


class MockPerfAnalyzerMethods(MockBase):
    """
    Mocks the subprocess module functions used in 
    model_analyzer/perf_analyzer/perf_analyzer.py
    Provides functions to check operation.
    """

    def __init__(self):
        self.mock_popen = MagicMock()
        self.mock_popen.pid = 10
        self.mock_popen.returncode = 0
        self.patcher_popen_stdout_read = patch(
            'model_analyzer.perf_analyzer.perf_analyzer.Popen',
            Mock(return_value=self.mock_popen))
        self.patcher_stdout = patch(
            'model_analyzer.perf_analyzer.perf_analyzer.STDOUT', MagicMock())
        self.patcher_pipe = patch(
            'model_analyzer.perf_analyzer.perf_analyzer.PIPE', MagicMock())
        super().__init__()

    def start(self):
        """
        Start the patchers
        """

        self.popen_stdout_read = self.patcher_popen_stdout_read.start()
        self.stdout_mock = self.patcher_stdout.start()
        self.pipe_mock = self.patcher_pipe.start()

    def _fill_patchers(self):
        """
        Fills patcher list
        """

        self._patchers.append(self.patcher_popen_stdout_read)
        self._patchers.append(self.patcher_stdout)
        self._patchers.append(self.patcher_pipe)

    def assert_perf_analyzer_run_as(self, cmd):
        """
        Checks that Popen was run with the given command.
        """

        self.popen_stdout_read.assert_called_with(cmd,
                                                  start_new_session=True,
                                                  stdout=self.pipe_mock,
                                                  stderr=self.stdout_mock,
                                                  encoding='utf-8')

    def set_perf_analyzer_result_string(self, output_string):
        """
        Sets the return value of mock_popen
        """

        self.mock_popen.stdout.read.return_value = output_string

    def get_perf_analyzer_popen_read_call_count(self):
        """
        Get perf_analyzer popen read count
        """

        return self.mock_popen.stdout.read.call_count

    def set_perf_analyzer_return_code(self, returncode):
        """
        Sets the returncode of Popen process
        """

        self.mock_popen.returncode = returncode

    def reset(self):
        """
        Resets the side effects
        and return values of the
        mocks in this module
        """

        self.mock_popen.stdout.read.side_effect = None
        self.mock_popen.stdout.read.return_value = None
