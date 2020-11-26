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
import sys
import os
from io import StringIO

from model_analyzer.output.file_writer import FileWriter
import test_result_collector as trc


class TestFileWriterMethods(trc.TestResultCollector):
    def test_write(self):
        test_handle = StringIO()
        writer = FileWriter(file_handle=test_handle)

        # Write test using create if not exist mode
        writer.write('test')

        # read file
        self.assertEqual(test_handle.getvalue(), 'test')

        # redirect stdout and create writer with no filename
        test_handle = StringIO()
        old_stdout = sys.stdout
        sys.stdout = test_handle
        writer = FileWriter()
        writer.write('test')
        sys.stdout.flush()
        sys.stdout = old_stdout

        self.assertEqual(test_handle.getvalue(), 'test')
        test_handle.close()

        # Check for malformed calls
        err_str = "Expected TritonModelAnalyzerException on malformed input."
        writer = FileWriter(file_handle=test_handle)
        with self.assertRaises(Exception, msg=err_str):
            writer.write('test')


if __name__ == '__main__':
    unittest.main()
