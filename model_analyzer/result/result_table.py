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

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from model_analyzer.constants import RESULT_TABLE_COLUMN_PADDING


class ResultTable:
    """
    A class that implements a generic table interface with headers rows
    """

    column_padding = RESULT_TABLE_COLUMN_PADDING

    def __init__(self, headers, title=None):
        """
        Parameters
        ----------
        headers : list of str
            Names of the columns of this table
        title : str
            Title of the table
        """

        self._headers = headers
        self._title = title
        self._column_widths = [
            len(header) + self.column_padding for header in headers
        ]
        self._rows = []

    def title(self):
        """
        Get table title

        Returns
        -------
        str
            Title of the table
        """
        return self._title

    def headers(self):
        """
        Returns
        -------
        list of str
            names of the columns of this table
        """

        return self._headers

    def size(self):
        """
        Returns
        -------
        int
            number of rows in this
            table
        """

        return len(self._rows)

    def column_widths(self):
        """
        Returns
        -------
        list of ints
            Current width in spaces of each column in table.
        """

        return self._column_widths

    def empty(self):
        """
        Returns
        -------
        True if this table has no data
        False if it does
        """

        return len(self._rows) == 0

    def insert_row_by_index(self, row, index=None):
        """
        Adds a row to the table. Handles wrapping.

        Parameters
        ----------
        row : list
            A row of data to add to the ResultTable

        Raises
        ------
        TritonModelAnalyzerException
            if there is a mismatch between the table headers
            and the row to be inserted.
        """

        if len(row) != len(self._headers):
            raise TritonModelAnalyzerException(
                f"Inserted row contains {len(row)} values."
                f"There are {len(self._headers)} provided headers.")
        if index is None:
            index = len(self._rows)
        self._rows.insert(index, row[:])

        for i in range(len(row)):
            self._column_widths[i] = max(
                len(str(row[i])) + self.column_padding, self._column_widths[i])

    def get_row_by_index(self, index):
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

    def remove_row_by_index(self, index):
        """
        Removes row at given index
        from the table

        Parameters
        ----------
        index : int
            The index of the row to be removed
        """

        if len(self._rows) == 0:
            raise TritonModelAnalyzerException(
                "Attempting to remove result from an empty ResultTable!")
        if index < 0 or index >= len(self._rows):
            raise TritonModelAnalyzerException(
                f"Index {index} out of range for remove_row_by_index")
        self._rows.pop(index)

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
        Converts a single row to its string representation
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
        Constructs a single cell of the table by either padding or truncating
        the value inside
        """

        diff = length - len(string)
        if diff >= 0:
            return string + (' ') * diff
        else:
            return string[:diff]
