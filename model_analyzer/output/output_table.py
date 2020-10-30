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
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class OutputTable:
    """
    A class that implements a 
    generic table interface 
    with headers rows
    """

    def __init__(self, headers):
        """
        Parameters
        ----------
        headers : list of str
            Names of the columns of this table
        """
        self._headers = headers
        self._rows = []

    def headers(self):
        """
        Returns
        -------
        list 
            names of the columns of this table
        """
        return self._headers

    def add_row(self, row, index=None):
        """
        Adds a row to the table

        Parameters
        ----------
        row : list of vals
            The contents of the row to be added
        index : int
            The index at which to add a row 
        """
        if len(row) != len(self._headers):
            raise TritonModelAnalyzerException(
                "Must provide a value for each existing column when adding a new row."
            )
        if index is None:
            index = len(self._rows)
        self._rows.insert(index, row)

    def add_column(self, column, index=None):
        """
        Adds a column to the table

        Parameters
        ----------
        column : list of vals
            Content of the column to be added
        index : int
            The index at which to add a column 
        """
        if len(column) != len(self._rows) + 1:
            raise TritonModelAnalyzerException(
                "Must provide a value for each existing row when adding a new column."
            )
        if index is None:
            index = len(self._headers)
        self._headers.insert(index, column[0])
        for i in range(len(self._rows)):
            self._rows[i].insert(index, column[i + 1])

    def get_row(self, index):
        """
        Returns the row at given index

        Parameters
        ----------
        index : int
            index of row to return
        
        Returns
        -------
        list of vals
            The contents of the desired column
        """
        if index < 0 or index >= len(self._rows):
            raise TritonModelAnalyzerException(
                f"Index {index} out of range for get_row")
        return self._rows[index]

    def get_column(self, index):
        """
        Returns the column at given index

        Parameters
        ----------
        index : int
            index of column to return
        
        Returns
        -------
        list of vals
            The contents of the desired column
        """
        if index < 0 or index >= len(self._headers):
            raise TritonModelAnalyzerException(
                f"Index {index} out of range for get_column")
        return [self._headers[index]] + [row[index] for row in self._rows]

    def remove_row_by_index(self, index):
        """
        Removes row at given index
        from the table

        Parameters
        ----------
        index : int
            The index of the row to be removed
        """
        if index < 0 or index >= len(self._rows):
            raise TritonModelAnalyzerException(
                f"Index {index} out of range for remove_row_by_index")
        self._rows.pop(index)

    def remove_column_by_index(self, index):
        """
        Removes column at given index
        from the table

        Parameters
        ----------
        index : int
            The index of the column to be removed
        """
        if index < 0 or index >= len(self._headers):
            raise TritonModelAnalyzerException(
                f"Index {index} out of range for remove_column_by_index")
        self._headers.pop(index)
        for row in self._rows:
            row.pop(index)

    def remove_column_by_header(self, header):
        """
        Removes column with given header
        from the table

        Parameters
        ----------
        header : str
            The name of the column to be removed
        """
        index = self._headers.index(header)
        self.remove_column_by_index(index)

    def to_formatted_string(self, column_width=None, separator=''):
        """
        Converts the table into its string representation
        making it easy to write by a writer

        Parameters
        ----------
        column_width : int
            The width in characters of each column in this
            table. If None, then each cell will be the width 
            of its contents.
        separator : str
            The string that will separate columns of a row in th
            table
        
        Returns
        -------
        str
            The formatted table as a string ready for writing
        """
        output_rows = []
        for row in [self._headers] + self._rows:
            output_rows.append(
                self._row_to_string(row, column_width, separator))
        return '\n'.join(output_rows)

    def _row_to_string(self, row, column_width, separator):
        """
        Converts a single row to 
        its string representation
        """
        return separator.join([
            self._pad_or_trunc(str(val), column_width)
            if column_width else str(val) for val in row
        ])

    def _pad_or_trunc(self, string, length):
        """
        Constructs a single cell of the table
        by either padding or truncating 
        the value inside
        """
        diff = length - len(string)
        if diff >= 0:
            return string + (' ') * diff
        else:
            return string[:diff]
