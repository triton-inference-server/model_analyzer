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

import argparse
import os
import importlib
import inspect
import sys
sys.path.insert(0, '../../')


def args():
    parser = argparse.ArgumentParser('test_counter')
    parser.add_argument('--path',
                        help='Path to use for counting the tests',
                        type=str)
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    number_of_tests = 0
    opt = args()
    path = opt.path

    for file_path in os.listdir(path):

        # All the test files start with "Test"
        if file_path.startswith('test_'):
            module_name = 'tests.' + file_path.split('.')[0]
            module = importlib.import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            for class_tuple in classes:
                class_name = class_tuple[0]
                class_object = class_tuple[1]

                # All the test classes start with "Test"
                if class_name.startswith('Test'):
                    methods = inspect.getmembers(class_object,
                                                 inspect.isroutine)
                    for method_tuple in methods:
                        method_name = method_tuple[0]
                        if method_name.startswith('test_'):
                            number_of_tests += 1

    # Print the number of tests
    print(number_of_tests)
