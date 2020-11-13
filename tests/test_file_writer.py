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
