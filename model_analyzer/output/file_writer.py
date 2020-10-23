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

from collections import defaultdict
from .table_writer import TableWriter
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class FileWriter(TableWriter):
    """
    Writes table to a file or stdout
    """
    def __init__(self, filename=None):
        """
        Parameters
        ----------
        filename : str
            The file or stream to write the output to. 
            If filename is None print to stdout.
        """
        self._filename = filename

    def write(self, table, write_mode=None):
        """
        Writes the output to a file or stdout

        Parameters
        ----------
        table : str
            The formatted table constructed from 
            recorded metrics as a string.
        
        write_mode : str
            How to open the file for writing
        
        Raises
        ------
        TritonModelAnalyzerException
            If there is an error or exception while writing
            the output.
        """
        allowed_write_modes = ['a', 'w', 'a+', 'w+', 'rw', 'ra']
        if self._filename:
            if write_mode in allowed_write_modes:
                try:
                    with open(self._filename, write_mode) as f:
                        f.write(table)
                except IOError as e:
                    f.close()
                    raise TritonModelAnalyzerException(e)
            else:
                raise TritonModelAnalyzerException(
                    f"Write mode must be one of {allowed_write_modes}")
        else:
            print(table, end='')
