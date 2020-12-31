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

import sys
sys.path.append("../common")

import unittest
import os

from .mocks.mock_io import MockIOMethods

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.output.file_writer import FileWriter
import test_result_collector as trc

TEST_FILENAME = 'test_filename'


class TestFileWriterMethods(trc.TestResultCollector):
    def setUp(self):
        self.io_mock = MockIOMethods()

    def test_write(self):
        # Create and use writer
        writer = FileWriter(filename=TEST_FILENAME)

        # Check for exception on open and write
        self.io_mock.raise_exception_on_open()
        err_str = "Expected TritonModelAnalyzerException on malformed input."
        with self.assertRaises(TritonModelAnalyzerException, msg=err_str):
            writer.write('test')
        self.io_mock.reset()

        self.io_mock.raise_exception_on_write()
        err_str = "Expected TritonModelAnalyzerException on malformed input."
        with self.assertRaises(TritonModelAnalyzerException, msg=err_str):
            writer.write('test')
        self.io_mock.reset()

        # Check mock call on successful write
        writer.write('test')
        self.io_mock.assert_write_called_with_args('test')

        # Perform checks for stdout
        writer = FileWriter()
        writer.write('test')
        self.io_mock.assert_print_called_with_args('test')

    def tearDown(self):
        self.io_mock.stop()


if __name__ == '__main__':
    unittest.main()
