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


class SearchDimensions:
    """
    Data class that holds one or more dimensions and associates each one to a key
    """

    def __init__(self):
        self._dimensions = []
        self._dimension_keys = []

    def add_dimensions(self, key, dimensions):
        """ 
        Add dimensions and associate them all with the given key

        Parameters
        ----------
        Key: str or int
            The key to associate the dimensions with

        Dimensions: List of SearchDimension
            Dimensions to add and associate with the key
        """
        for dim in dimensions:
            self._dimensions.append(dim)
            self._dimension_keys.append(key)

    def get_values_for_coordinate(self, coordinate):
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
        vals = {}
        for i, v in enumerate(coordinate):
            key = self._dimension_keys[i]
            if key not in vals:
                vals[key] = {}

            dim = self._dimensions[i]
            name = dim.get_name()
            val = dim.get_value_at_idx(v)
            vals[key][name] = val

        return vals

    def __iter__(self):
        return iter(self._dimensions)

    def __len__(self):
        return len(self._dimensions)

    def __getitem__(self, index):
        return self._dimensions[index]
