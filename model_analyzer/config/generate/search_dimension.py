# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import sys


class SearchDimension:
    """
    Defines a single dimension to search, and how the values
    of that dimension grow
    """

    DIMENSION_TYPE_LINEAR = 0
    DIMENSION_TYPE_EXPONENTIAL = 1
    DIMENSION_NO_MAX = sys.maxsize

    def __init__(self,
                 name: str,
                 type: int,
                 min: int = 0,
                 max: int = DIMENSION_NO_MAX):
        """
        Parameters
        ----------
        name: str
        type: enum
            Enum indicating how the values of this dimension grow
        min: int
            The minimum index for this search dimension. If unspecified, min is 0
        max: int
            The maximum index for this search dimension. If unspecified, then there is no max

        """
        self._name = name
        self._type = type
        self._min = min
        self._max = max

    def get_min_idx(self) -> int:
        """ Return the minimum index for this dimension"""
        return self._min

    def get_max_idx(self) -> int:
        """ Return the maximum index for this dimension"""
        return self._max

    def get_name(self) -> str:
        """ Return the name for this dimension """
        return self._name

    def get_value_at_idx(self, idx: int) -> int:
        """ Return the value of the dimension at the given index """
        if idx < self._min or idx > self._max:
            raise IndexError(
                f"Index {idx} is out of range for search dimension {self._name}"
            )

        if self._type == SearchDimension.DIMENSION_TYPE_LINEAR:
            return idx + 1
        elif self._type == SearchDimension.DIMENSION_TYPE_EXPONENTIAL:
            return int(math.pow(2, idx))
        else:
            raise Exception(f"Unknown type {self._type}")
