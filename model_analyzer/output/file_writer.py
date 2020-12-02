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

from collections import defaultdict
from .output_writer import OutputWriter
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class FileWriter(OutputWriter):
    """
    Writes table to a file or stdout
    """

    def __init__(self, file_handle=None):
        """
        Parameters
        ----------
        filename : File
            The file or stream pointer to write the output to.
            Writer to stdout if file_handle is None
        """
        self._file_handle = file_handle

    def write(self, out):
        """
        Writes the output to a file or stdout

        Parameters
        ----------
        out : str
            The string to be written to the
            file or stdout

        Raises
        ------
        TritonModelAnalyzerException
            If there is an error or exception while writing
            the output.
        """
        if self._file_handle:
            try:
                self._file_handle.write(out)
            except IOError as e:
                raise TritonModelAnalyzerException(e)
        else:
            print(out, end='')
