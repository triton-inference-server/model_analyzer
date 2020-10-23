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

import unittest
import sys
import os
from model_analyzer.output.file_writer import FileWriter


class TestFileWriterMethods(unittest.TestCase):
    def test_write(self):
        filename = 'test_file'
        writer = FileWriter(filename=filename)

        # Write test using create if not exist mode
        writer.write('test', write_mode='w+')

        # open and read file
        with open(filename, 'r') as f:
            content = f.read()

        self.assertEqual(content, 'test')

        # redirect stdout and create writer with no filename
        old_stdout = sys.stdout
        sys.stdout = open('test_file_stdout', 'w')
        writer = FileWriter()
        writer.write('test')
        sys.stdout.flush()
        sys.stdout.close()
        sys.stdout = old_stdout

        with open('test_file_stdout', 'r') as f:
            content = f.read()

        self.assertEqual(content, 'test')

        # Check for malformed calls
        err_str = "Expected TritonModelAnalyzerException on malformed input."
        writer = FileWriter('file_that_does_not_exist')
        with self.assertRaises(Exception, msg=err_str):
            writer.write('test', write_mode='rw')
        with self.assertRaises(Exception, msg=err_str):
            writer.write('test', write_mode='x+')

        # Remove test files
        if os.path.exists("test_file"):
            os.remove("test_file")
        if os.path.exists("test_file_stdout"):
            os.remove("test_file_stdout")


if __name__ == '__main__':
    unittest.main()