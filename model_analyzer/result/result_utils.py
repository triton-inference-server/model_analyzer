# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
"""
A module for various functions
needed across the result module
"""


def average_list(row_list):
    """
    Averages a 2d list over the rows
    """

    if not row_list:
        return row_list
    else:
        N = len(row_list)
        d = len(row_list[0])
        avg = [0 for _ in range(d)]
        for i in range(d):
            avg[i] = (sum([row_list[j][i] for j in range(1, N)],
                          start=row_list[0][i]) * 1.0) / N
        return avg
