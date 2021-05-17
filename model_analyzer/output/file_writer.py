# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

from .output_writer import OutputWriter
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class FileWriter(OutputWriter):
    """
    Writes table to a file or stdout
    """

    def __init__(self, filename=None):
        """
        Parameters
        ----------
        filename : File
            The full path to the file or stream to write the output to.
            Writes to stdout if filename is None
        """

        self._filename = filename

    def write(self, out, append=False):
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

        write_mode = 'a+' if append else 'w+'
        if self._filename:
            try:
                with open(self._filename, write_mode) as f:
                    f.write(out)
            except OSError as e:
                raise TritonModelAnalyzerException(e)
        else:
            print(out, end='')
