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

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

COLUMN_PADDING = 2


class OutputTable:
    """
    A class that implements a 
    generic table interface 
    with headers rows
    """

    column_padding = COLUMN_PADDING

    def __init__(self, headers):
        """
        Parameters
        ----------
        headers : list of str
            Names of the columns of this table
        """

        self._headers = headers[:]
        self._rows = []
        self._column_widths = [
            len(header) + self.column_padding for header in headers
        ]

    def headers(self):
        """
        Returns
        -------
        list of str
            names of the columns of this table
        """

        return self._headers

    def column_widths(self):
        """
        Returns
        -------
        list of ints
            Current width in spaces of 
            each column in table.
        """

        return self._column_widths

    def add_row(self, row, index=None):
        """
        Adds a row to the table. Handles
        wrapping.

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
        self._rows.insert(index, row[:])

        for i in range(len(row)):
            self._column_widths[i] = max(
                len(str(row[i])) + self.column_padding, self._column_widths[i])

    def add_column(self, column, index=None):
        """
        Adds a column to the table. 
        
        **Note : column[0] is assumed
        to be the column header.

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
        column_width = max(
            [len(str(val)) + self.column_padding for val in column])
        self._column_widths.insert(index, column_width)

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

    def set_column_width_by_index(self, index, width):
        """
        Allows setting the column width in the string
        representation of the table

        Parameters
        ----------
        header : str
            Name of the column whose width to adjust
        width : int
            The new width in spaces
        """

        if index < 0 or index >= len(self._column_widths):
            raise TritonModelAnalyzerException(
                f"Index {index} out of range for set_column_width_by_index")
        self._column_widths[index] = width

    def set_column_width_by_header(self, header, width):
        """
        Allows setting the column width in the string
        representation of the table

        Parameters
        ----------
        header : str
            Name of the column whose width to adjust
        width : int
            The new width in spaces
        """

        try:
            index = self._headers.index(header)
        except ValueError:
            raise TritonModelAnalyzerException(
                f"{header} not present in table")
        self._column_widths[index] = width

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
        self._column_widths.pop(index)
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

        try:
            index = self._headers.index(header)
        except ValueError:
            raise TritonModelAnalyzerException(
                f"{header} not present in table")
        self.remove_column_by_index(index)

    def to_formatted_string(self, separator='', ignore_widths=False):
        """
        Converts the table into its string representation
        making it easy to write by a writer

        Parameters
        ----------
        separator : str
            The string that will separate columns of a row in th
            table
        ignore_widths : bool
            Each cell is as wide as its content. Useful
            for csv format.

        Returns
        -------
        str
            The formatted table as a string ready for writing
        """

        output_rows = []
        for row in [self._headers] + self._rows:
            output_rows.append(
                self._row_to_string(row, separator, ignore_widths))
        return '\n'.join(output_rows)

    def _row_to_string(self, row, separator, ignore_widths):
        """
        Converts a single row to 
        its string representation
        """

        if ignore_widths:
            return separator.join([str(row[j]) for j in range(len(row))])
        else:
            return separator.join([
                self._pad_or_trunc(str(row[j]), self._column_widths[j])
                for j in range(len(row))
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
