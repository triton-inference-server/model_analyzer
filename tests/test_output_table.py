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

from model_analyzer.output.output_table import OutputTable

TEST_HEADERS = [f"Column {i}" for i in range(10)]
TEST_ROWS = [[f"value {i}{j}" for j in range(10)] for i in range(10)]
TEST_TABLE_STR = (
    "Column 0    Column 1    Column 2    Column 3    Column 4    Column 5    Column 6    Column 9    \n"
    "value 00    value 01    value 02    value 03    value 04    value 05    value 06    value 09    \n"
    "value 10    value 11    value 12    value 13    value 14    value 15    value 16    value 19    \n"
    "value 20    value 21    value 22    value 23    value 24    value 25    value 26    value 29    \n"
    "value 30    value 31    value 32    value 33    value 34    value 35    value 36    value 39    \n"
    "value 40    value 41    value 42    value 43    value 44    value 45    value 46    value 49    \n"
    "value 50    value 51    value 52    value 53    value 54    value 55    value 56    value 59    \n"
    "value 60    value 61    value 62    value 63    value 64    value 65    value 66    value 69    \n"
    "value 70    value 71    value 72    value 73    value 74    value 75    value 76    value 79    \n"
    "value 80    value 81    value 82    value 83    value 84    value 85    value 86    value 89    \n"
    "value 90    value 91    value 92    value 93    value 94    value 95    value 96    value 99    "
)


class TestOutputTableMethods(unittest.TestCase):
    def setUp(self):
        pass

    def test_create_headers(self):
        table = OutputTable(headers=["Column 0"])
        self.assertEqual(table.headers(), ["Column 0"])

    def test_add_get_methods(self):
        table = OutputTable(headers=["Column 0"])

        # add/get single row/col
        table.add_row(["value 0,0"])
        self.assertEqual(table.get_row(index=0), ["value 0,0"])

        table.add_column(["Column 1", "value 0,1"])
        self.assertEqual(table.headers(), ["Column 0", "Column 1"])
        self.assertEqual(table.get_row(index=0), ["value 0,0", "value 0,1"])
        self.assertEqual(table.get_column(index=1), ["Column 1", "value 0,1"])

        # add/get row and column by index
        table.add_row(["value -1,0", "value -1,1"], index=0)
        self.assertEqual(table.get_row(index=0), ["value -1,0", "value -1,1"])
        self.assertEqual(table.get_column(index=0),
                         ["Column 0", "value -1,0", "value 0,0"])
        self.assertEqual(table.get_column(index=1),
                         ["Column 1", "value -1,1", "value 0,1"])

        table.add_column(["Column 0.5", "value -1,0.5", "value 0,0.5"],
                         index=1)
        self.assertEqual(table.get_column(index=1),
                         ["Column 0.5", "value -1,0.5", "value 0,0.5"])
        self.assertEqual(table.get_row(index=0),
                         ["value -1,0", "value -1,0.5", "value -1,1"])
        self.assertEqual(table.get_row(index=1),
                         ["value 0,0", "value 0,0.5", "value 0,1"])

    def test_remove_methods(self):
        table = OutputTable(headers=TEST_HEADERS)
        for row in TEST_ROWS:
            table.add_row(row)

        # Pick a row
        row_idx = 4
        self.assertEqual(table.get_row(index=row_idx),
                         [f"value 4{j}" for j in range(10)])

        # remove row and check that the next one is in its place
        table.remove_row_by_index(index=row_idx)
        self.assertEqual(table.get_row(index=row_idx),
                         [f"value 5{j}" for j in range(10)])

        table = OutputTable(headers=TEST_HEADERS)
        for row in TEST_ROWS:
            table.add_row(row)

        # Pick column, remove and check that is replaced
        col_idx = 7
        self.assertEqual(table.get_column(index=col_idx),
                         ["Column 7"] + [f"value {i}7" for i in range(10)])

        table.remove_column_by_index(index=col_idx)
        self.assertEqual(table.get_column(index=col_idx),
                         ["Column 8"] + [f"value {i}8" for i in range(10)])

        table.remove_column_by_header(header="Column 8")
        self.assertEqual(table.get_column(index=col_idx),
                         ["Column 9"] + [f"value {i}9" for i in range(10)])

    def test_to_formatted_string(self):
        table = OutputTable(headers=TEST_HEADERS)
        for row in TEST_ROWS:
            table.add_row(row)
        self.assertEqual(table.to_formatted_string(), TEST_TABLE_STR)


if __name__ == '__main__':
    unittest.main()