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

import unittest

from model_analyzer.output.output_table import OutputTable
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from .common import test_result_collector as trc

TEST_HEADERS = [f"Column {i}" for i in range(10)]
TEST_COLUMN_WIDTH = 12
TEST_WIDTHS = [TEST_COLUMN_WIDTH for i in range(10)]
TEST_ROWS = [[f"value {i}{j}" for j in range(10)] for i in range(10)]
TEST_TABLE_STR = (
    "Column 0  Column 1  Column 2  Column 3  Column 4  Column 5  Column 6  Column 7  Column 8  Column 9  \n"
    "value 00  value 01  value 02  value 03  value 04  value 05  value 06  value 07  value 08  value 09  \n"
    "value 10  value 11  value 12  value 13  value 14  value 15  value 16  value 17  value 18  value 19  \n"
    "value 20  value 21  value 22  value 23  value 24  value 25  value 26  value 27  value 28  value 29  \n"
    "value 30  value 31  value 32  value 33  value 34  value 35  value 36  value 37  value 38  value 39  \n"
    "value 40  value 41  value 42  value 43  value 44  value 45  value 46  value 47  value 48  value 49  \n"
    "value 50  value 51  value 52  value 53  value 54  value 55  value 56  value 57  value 58  value 59  \n"
    "value 60  value 61  value 62  value 63  value 64  value 65  value 66  value 67  value 68  value 69  \n"
    "value 70  value 71  value 72  value 73  value 74  value 75  value 76  value 77  value 78  value 79  \n"
    "value 80  value 81  value 82  value 83  value 84  value 85  value 86  value 87  value 88  value 89  \n"
    "value 90  value 91  value 92  value 93  value 94  value 95  value 96  value 97  value 98  value 99  "
)
TEST_CSV_STR = (
    "Column 0,Column 1,Column 2,Column 3,Column 4,Column 5,Column 6,Column 7,Column 8,Column 9\n"
    "value 00,value 01,value 02,value 03,value 04,value 05,value 06,value 07,value 08,value 09\n"
    "value 10,value 11,value 12,value 13,value 14,value 15,value 16,value 17,value 18,value 19\n"
    "value 20,value 21,value 22,value 23,value 24,value 25,value 26,value 27,value 28,value 29\n"
    "value 30,value 31,value 32,value 33,value 34,value 35,value 36,value 37,value 38,value 39\n"
    "value 40,value 41,value 42,value 43,value 44,value 45,value 46,value 47,value 48,value 49\n"
    "value 50,value 51,value 52,value 53,value 54,value 55,value 56,value 57,value 58,value 59\n"
    "value 60,value 61,value 62,value 63,value 64,value 65,value 66,value 67,value 68,value 69\n"
    "value 70,value 71,value 72,value 73,value 74,value 75,value 76,value 77,value 78,value 79\n"
    "value 80,value 81,value 82,value 83,value 84,value 85,value 86,value 87,value 88,value 89\n"
    "value 90,value 91,value 92,value 93,value 94,value 95,value 96,value 97,value 98,value 99"
)


class TestOutputTableMethods(trc.TestResultCollector):
    def test_create_headers(self):
        table = OutputTable(headers=["Column 0"])
        self.assertEqual(table.headers(), ["Column 0"])
        self.assertEqual(table.column_widths(),
                         [len("Column 0") + OutputTable.column_padding])

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

    def test_set_width_methods(self):
        table = OutputTable(headers=TEST_HEADERS)
        for row in TEST_ROWS:
            table.add_row(row[:])
        self.assertEqual(table._column_widths[0], 10)

        # Now set the width by header
        table.set_column_width_by_header(header='Column 0', width=10)
        self.assertEqual(table.column_widths()[0], 10)
        table.set_column_width_by_header(header='Column 4', width=2)
        self.assertEqual(table.column_widths()[4], 2)
        with self.assertRaises(
                TritonModelAnalyzerException,
                msg="Expected invalid header to raise Exception"):
            table.set_column_width_by_header(header='Column NOT PRESENT',
                                             width=2)

        # Now set the width by index
        table.set_column_width_by_index(index=3, width=10)
        self.assertEqual(table.column_widths()[3], 10)
        table.set_column_width_by_index(index=7, width=2)
        self.assertEqual(table.column_widths()[7], 2)
        with self.assertRaises(
                TritonModelAnalyzerException,
                msg="Expected invalid index to raise Exception"):
            table.set_column_width_by_index(index=12, width=2)

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

        table = OutputTable(headers=TEST_HEADERS)
        for row in TEST_ROWS:
            table.add_row(row)
        self.assertEqual(
            table.to_formatted_string(separator=',', ignore_widths=True),
            TEST_CSV_STR)


if __name__ == '__main__':
    unittest.main()
