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


class SearchConfig:
    """
    Defines all dimensions to search
    """

    def __init__(self, dimensions):
        self._dimensions = dimensions

    def get_num_dimensions(self):
        """ Returns the number of dimensions in this search """
        return len(self._dimensions)

    def get_dimension(self, idx):
        """ Returns the SearchDimension at the given index """
        return self._dimensions[idx]

    def get_min_dimension(self):
        """ 
        Returns a coordinate cooresponding to the minimum of all SearchDimensions
        """
        coordinate = []
        for dimension in self._dimensions:
            coordinate.append(dimension.get_min_idx())
        return coordinate