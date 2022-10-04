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

from .search_dimension import SearchDimension
from .coordinate import Coordinate

from typing import Any, Iterator, List, Dict


class SearchDimensions:
    """
    Data class that holds one or more dimensions and associates each one to a key
    """

    def __init__(self) -> None:
        self._dimensions: List[SearchDimension] = []
        self._dimension_keys: List[Any] = []

    def add_dimensions(self, key: Any,
                       dimensions: List[SearchDimension]) -> None:
        """ 
        Add dimensions and associate them all with the given key

        Parameters
        ----------
        Key: int
            The key to associate the dimensions with

        Dimensions: List of SearchDimension
            Dimensions to add and associate with the key
        """
        for dim in dimensions:
            self._dimensions.append(dim)
            self._dimension_keys.append(key)

    def get_values_for_coordinate(
            self, coordinate: Coordinate) -> Dict[Any, Dict[str, Any]]:
        """
        Given a Coordinate, return all dimension_name:dimension_value pairs associated with 
        that coordinate, organized by the dimension's key
        
        Parameters
        ----------
        
        coordinate: Coordinate
            The coordinate to get values for 

        Returns: Dict of Dicts
            ret[key][SearchDimension name] = value

        """
        vals: Dict[Any, Dict[str, Any]] = {}
        for i, v in enumerate(coordinate):
            key = self._dimension_keys[i]
            if key not in vals:
                vals[key] = {}

            dim = self._dimensions[i]
            name = dim.get_name()
            val = dim.get_value_at_idx(v)
            vals[key][name] = val

        return vals

    def __iter__(self) -> Iterator:
        return iter(self._dimensions)

    def __len__(self) -> int:
        return len(self._dimensions)

    def __getitem__(self, index: int) -> SearchDimension:
        return self._dimensions[index]
