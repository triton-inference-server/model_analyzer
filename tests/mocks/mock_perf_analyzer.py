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
        self.patcher_check_output = patch(
            'model_analyzer.perf_analyzer.perf_analyzer.check_output')
        self.patcher_stdout = patch(
            'model_analyzer.perf_analyzer.perf_analyzer.STDOUT', MagicMock())
        self.patcher_called_process_error = patch(
            'model_analyzer.perf_analyzer.perf_analyzer.CalledProcessError',
            MockCalledProcessError)
        super().__init__()

    def start(self):
        """
        Start the patchers
        """

        self.check_output_mock = self.patcher_check_output.start()
        self.stdout_mock = self.patcher_stdout.start()
        self.called_process_error_mock = self.patcher_called_process_error.start(
        )

    def _fill_patchers(self):
        """
        Fills patcher list
        """

        self._patchers.append(self.patcher_check_output)
        self._patchers.append(self.patcher_stdout)
        self._patchers.append(self.patcher_called_process_error)

    def assert_perf_analyzer_run_as(self, cmd):
        """
        Checks that subprocess.check_output was run
        with the given command.
        """

        self.check_output_mock.assert_called_with(cmd,
                                                  start_new_session=True,
                                                  stderr=self.stdout_mock,
                                                  encoding='utf-8')

    def raise_exception_on_run(self):
        """
        Raises a CalledProcessError on call
        check_output
        """

        self.check_output_mock.side_effect = self.called_process_error_mock

    def set_perf_analyzer_result_string(self, output_string):
        """
        Sets the return value of subprocess.check_output
        """

        self.check_output_mock.return_value = output_string

    def reset(self):
        """
        Resets the side effects
        and return values of the
        mocks in this module
        """

        self.check_output_mock.side_effect = None
        self.check_output_mock.return_value = None
