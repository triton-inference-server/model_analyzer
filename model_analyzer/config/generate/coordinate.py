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

from copy import deepcopy
from typing import Iterator, Any, Union, List


class Coordinate:
    """
    Class to define a coordinate in n-dimension space
    """

    def __init__(self, val: Union['Coordinate', List[int]]):
        """
        val: list
            List of floats or integers corresponding to the location in space
        """
        if isinstance(val, Coordinate):
            val = val._values

        self._values: List[int] = deepcopy(val)

    def __getitem__(self, idx: int) -> int:
        return self._values[idx]

    def __setitem__(self, idx: int, item: int) -> None:
        self._values[idx] = item

    def __len__(self) -> int:
        return len(self._values)

    def __add__(self, other: Any) -> 'Coordinate':
        if type(other) == Coordinate:
            return self._add_coordinate(other)
        elif type(other) == int or type(other) == float:
            return self._add_number(other)
        else:
            raise Exception("Unhandled addition type")

    def __sub__(self, other: Any) -> 'Coordinate':
        if type(other) == Coordinate:
            return self._sub_coordinate(other)
        elif type(other) == int or type(other) == float:
            return self._sub_number(other)
        else:
            raise Exception("Unhandled subtraction type")

    def __truediv__(self, other: Any) -> 'Coordinate':
        if type(other) == int or type(other) == float:
            return self._div_number(other)
        else:
            raise Exception("Unhandled division type")

    def __mul__(self, other: Any) -> 'Coordinate':
        if type(other) == int or type(other) == float:
            return self._mul_number(other)
        else:
            raise Exception("Unhandled mul type")

    def __eq__(self, other: Any) -> bool:
        for i, v in enumerate(self._values):
            if v != other[i]:
                return False
        return True

    def __lt__(self, other: Any) -> bool:
        for i, v in enumerate(self._values):
            if v != other[i]:
                return v < other[i]
        return False

    def round(self) -> None:
        """ Rounds the coordinate in-place """
        for i, _ in enumerate(self._values):
            self._values[i] = round(self._values[i])

    def _add_coordinate(self, other: Any) -> 'Coordinate':
        ret = Coordinate(self._values)
        for i, v in enumerate(self._values):
            ret[i] = v + other[i]
        return ret

    def _add_number(self, other: Any) -> 'Coordinate':
        ret = Coordinate(self._values)
        for i, v in enumerate(self._values):
            ret[i] = v + other
        return ret

    def _sub_coordinate(self, other: Any) -> 'Coordinate':
        ret = Coordinate(self._values)
        for i, v in enumerate(self._values):
            ret[i] = v - other[i]
        return ret

    def _sub_number(self, other: Any) -> 'Coordinate':
        ret = Coordinate(self._values)
        for i, v in enumerate(self._values):
            ret[i] = v - other
        return ret

    def _mul_number(self, other: Any) -> 'Coordinate':
        ret = Coordinate(self._values)
        for i, v in enumerate(self._values):
            ret[i] = v * other
        return ret

    def _div_number(self, other: Any) -> 'Coordinate':
        ret = Coordinate(self._values)
        for i, v in enumerate(self._values):
            ret[i] = v / other
        return ret

    def __iter__(self) -> Iterator:
        self._idx = 0
        return self

    def __next__(self) -> int:
        if self._idx < len(self._values):
            val = self._values[self._idx]
            self._idx += 1
            return val
        raise StopIteration

    def __str__(self) -> str:
        return str(self._values)

    def __repr__(self) -> str:
        return repr(f"Coordinate({self._values})")
