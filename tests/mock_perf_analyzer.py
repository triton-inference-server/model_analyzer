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
        self.check_output_mock = self.patcher_check_output.start()
        self.stdout_mock = self.patcher_stdout.start()

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

    def set_perf_analyzer_result_string(self, output_string):
        """
        Sets the return value of subprocess.check_output
        """

        self.check_output_mock.return_value = output_string
