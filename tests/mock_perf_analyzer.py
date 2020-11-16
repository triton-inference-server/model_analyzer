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

import subprocess
from unittest.mock import patch, Mock, MagicMock


class MockCalledProcessError(Exception):
    """
    A mock of subprocess.CalledProcessError
    """

    def __init__(self):
        self.returncode = 1
        self.cmd = ["dummy command"]


class MockPerfAnalyzerMethods:
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
        self.check_output_mock = self.patcher_check_output.start()
        self.stdout_mock = self.patcher_stdout.start()
        self.called_process_error_mock = self.patcher_called_process_error.start(
        )

    def stop(self):
        """
        Destroy the mocked classes and
        functions
        """

        self.patcher_check_output.stop()
        self.patcher_stdout.stop()

    def assert_perf_analyzer_run_as(self, cmd):
        """
        Checks that subprocess.check_output was run
        with the given command.
        """

        self.check_output_mock.assert_called_with(cmd,
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
