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

from model_analyzer.config.generate.search_dimension import SearchDimension
from .search_dimensions import SearchDimensions
from typing import List, Optional


class NeighborhoodConfig:
    """
    Defines the configuration for a Neighborhood object
    """

    def __init__(self, dimensions: SearchDimensions, radius: int,
                 min_initialized: int):
        """
        Parameters
        ----------
        dimensions: SearchDimensions
        radius: int
            All points within distance=radius from a location will be in
            its neighborhood
        min_initialized: int
            Minimum number of initialized values in a neighborhood
            before a step can be taken
        """
        self._dimensions = dimensions
        self._radius = radius
        self._min_initialized = min_initialized

    def get_num_dimensions(self) -> int:
        """ Returns the number of dimensions in this search """
        return len(self._dimensions)

    def get_dimensions(self) -> SearchDimensions:
        """ Returns the SearchDimensions """
        return self._dimensions

    def get_dimension(self, idx: int) -> SearchDimension:
        """ Returns the SearchDimension at the given index """
        return self._dimensions[idx]

    def get_min_indexes(self) -> List[int]:
        """ 
        Returns a list cooresponding to the minimum index of all SearchDimensions
        """
        min_indexes = []
        for dimension in self._dimensions:
            min_indexes.append(dimension.get_min_idx())
        return min_indexes

    def get_min_initialized(self) -> int:
        """ 
        Returns the minimun number of initialized coordinates needed
        in a neighborhood before a step can be taken
        """
        return self._min_initialized

    def get_radius(self) -> int:
        """ Returns the base radius of a neighborhood """
        return self._radius


class SearchConfig(NeighborhoodConfig):
    """
    Defines all dimensions to search
    """

    def __init__(self, dimensions: SearchDimensions, radius: int,
                 min_initialized: int):
        """
        Parameters
        ----------
        dimensions: SearchDimensions
        radius: int
            All points within distance=radius from a location will be in
            each neighborhood
        min_initialized: int
            Minimum number of initialized values in a neighborhood
            before a step can be taken
        
        """
        super().__init__(dimensions=dimensions,
                         radius=radius,
                         min_initialized=min_initialized)

    def get_neighborhood_config(self,
                                radius: Optional[int] = None
                               ) -> NeighborhoodConfig:
        """
        Return a NeighborhoodConfig with an optional override to the radius
        """
        radius_to_use = radius if radius is not None else self.get_radius()

        return NeighborhoodConfig(dimensions=self._dimensions,
                                  radius=radius_to_use,
                                  min_initialized=self._min_initialized)
