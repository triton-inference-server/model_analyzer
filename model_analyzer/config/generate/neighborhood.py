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
from itertools import product
from copy import deepcopy

from typing import List, Tuple, Dict, Optional

from model_analyzer.config.generate.coordinate import Coordinate
from model_analyzer.config.generate.coordinate_data import CoordinateData
from model_analyzer.config.generate.search_config import NeighborhoodConfig
from model_analyzer.result.run_config_measurement import RunConfigMeasurement


class Neighborhood:
    """
    Defines and operates on a set of coordinates within a radius around
    a 'home' coordinate
    """

    def __init__(self, neighborhood_config: NeighborhoodConfig,
                 home_coordinate: Coordinate):
        """
        Parameters
        ----------
        neighborhood_config: 
            NeighborhoodConfig object
        home_coordinate: 
            Coordinate object to center the neighborhood around
        """
        assert type(neighborhood_config) == NeighborhoodConfig

        self._config = neighborhood_config
        self._home_coordinate = home_coordinate
        self._coordinate_data = CoordinateData()

        self._radius = self._config.get_radius()
        self._neighborhood = self._create_neighborhood()

    @property
    def coordinate_data(self):
        return self._coordinate_data

    @classmethod
    def calc_distance(cls, coordinate1: Coordinate,
                      coordinate2: Coordinate) -> float:
        """ 
        Return the euclidean distance between two coordinates
        """

        distance = 0.0
        for i, _ in enumerate(coordinate1):
            diff = coordinate1[i] - coordinate2[i]
            distance += math.pow(diff, 2)
        distance = math.sqrt(distance)
        return distance

    def enough_coordinates_initialized(self) -> bool:
        """
        Returns true if enough coordinates inside of the neighborhood
        have been initialized. Else false
        """
        min_initialized = self._config.get_min_initialized()
        num_initialized = len(self._get_visited_coordinates())
        return num_initialized >= min_initialized

    def calculate_new_coordinate(self,
                                 magnitude: int,
                                 enable_clipping: bool = True,
                                 clip_value: int = 2) -> Coordinate:
        """
        Based on the measurements in the neighborhood, determine where
        the next location should be

        Parameters
        ----------
        magnitude
            How large of a step to take
        disable_clipping
            Determines whether or not to clip the final step vector.

        Returns
        -------
        new_coordinate
            The new coordinate computed based on the neighborhood measurements.
        """
        step_vector = self._get_step_vector() * magnitude
        step_vector.round()

        if enable_clipping:
            step_vector = self._clip_coordinate_values(coordinate=step_vector,
                                                       clip_value=clip_value)

        tmp_new_coordinate = self._home_coordinate + step_vector
        new_coordinate = self._clamp_coordinate_to_bounds(tmp_new_coordinate)
        return new_coordinate

    def _clip_coordinate_values(self, coordinate: Coordinate,
                                clip_value: int) -> Coordinate:
        """
        Clip the coordinate values to be within the interval range of
        [-clip_value, clip_value].
        """
        assert clip_value >= 0, "clip_value must be non-negative number."

        for i in range(len(coordinate)):
            coordinate[i] = max(-clip_value, min(coordinate[i], clip_value))
        return coordinate

    def pick_coordinate_to_initialize(self) -> Optional[Coordinate]:
        """
        Based on the initialized coordinate values, pick an unvisited
        coordinate to initialize next.
        """
        covered_values_per_dimension = self._get_covered_values_per_dimension()

        max_num_uncovered = -1
        best_coordinate = None
        for coordinate in self._neighborhood:
            if not self._is_coordinate_visited(coordinate):
                num_uncovered = self._get_num_uncovered_values(
                    coordinate, covered_values_per_dimension)

                if num_uncovered > max_num_uncovered:
                    max_num_uncovered = num_uncovered
                    best_coordinate = coordinate

        return best_coordinate

    def get_nearest_neighbor(self,
                             coordinate_in: Coordinate) -> Optional[Coordinate]:
        """
        Find the nearest coordinate to the `coordinate_in` among the
        coordinates within the current neighborhood.
        """
        min_distance = float('inf')
        nearest_neighbor = None

        for coordinate in self._neighborhood:
            distance = Neighborhood.calc_distance(coordinate, coordinate_in)
            if distance < min_distance:
                nearest_neighbor = coordinate
                min_distance = distance

        return nearest_neighbor

    def _create_neighborhood(self) -> List[Coordinate]:

        neighborhood = []
        potential_neighborhood = self._get_potential_neighborhood(
            self._home_coordinate, self._radius)

        for potential_coordinate in potential_neighborhood:
            distance = Neighborhood.calc_distance(self._home_coordinate,
                                                  potential_coordinate)

            if distance <= self._radius:
                neighborhood.append(potential_coordinate)

        return neighborhood

    def _get_potential_neighborhood(self, coordinate: Coordinate,
                                    radius: int) -> List[Coordinate]:
        bounds = self._get_bounds(coordinate, radius)
        potential_values = self._enumerate_all_values_in_bounds(bounds)
        return [Coordinate(x) for x in potential_values]

    def _get_bounds(self, coordinate: Coordinate,
                    radius: int) -> List[List[int]]:
        bounds = []
        for i in range(self._config.get_num_dimensions()):
            dimension = self._config.get_dimension(i)

            lower_bound = max(dimension.get_min_idx(), coordinate[i] - radius)
            upper_bound = min(dimension.get_max_idx(),
                              coordinate[i] + radius + 1)
            bounds.append([lower_bound, upper_bound])
        return bounds

    def _enumerate_all_values_in_bounds(
            self, bounds: List[List[int]]) -> List[List[int]]:
        possible_index_values = []
        for bound in bounds:
            possible_index_values.append(list(range(bound[0], bound[1])))

        tuples = list(product(*possible_index_values))
        return [list(x) for x in tuples]

    def _get_visited_coordinates(self) -> List[Coordinate]:
        """
        Returns the list of coordinates in the neighborhood that have been
        visited (except the home coordinate).
        """
        visited_coordinates = []
        for coordinate in self._neighborhood:
            if coordinate != self._home_coordinate \
                    and self._is_coordinate_visited(coordinate):
                visited_coordinates.append(deepcopy(coordinate))
        return visited_coordinates

    def _get_step_vector(self) -> Coordinate:
        """
        Calculate a vector that indicates a direction to step from the
        home coordinate (current center).

        Returns
        -------
        step_vector
            a coordinate that tells the direction to move.
        """
        step_vector = Coordinate([0] * self._config.get_num_dimensions())

        vectors, measurements = self._compile_neighborhood_measurements()
        home_measurement = self._coordinate_data.get_measurement(
            coordinate=self._home_coordinate)

        for vector, measurement in zip(vectors, measurements):
            if measurement:
                weight = home_measurement.compare_measurements(measurement)
            else:
                # Minimize the effect of None measurement on the step vector.
                weight = 0.0

            step_vector += vector * weight

        step_vector /= len(vectors)
        return step_vector

    def _compile_neighborhood_measurements(
            self) -> Tuple[List[Coordinate], List[RunConfigMeasurement]]:
        """
        Gather all the vectors (directions from the home coordinate)
        and their corresponding measurements.

        Returns
        -------
        (vectors, measurements)
            collection of vectors and their measurements.
        """
        visited_coordinates = self._get_visited_coordinates()

        vectors = []
        measurements = []
        for coordinate in visited_coordinates:
            measurement = self._coordinate_data.get_measurement(coordinate)
            vectors.append(coordinate - self._home_coordinate)
            measurements.append(measurement)
        return vectors, measurements

    def _is_coordinate_visited(self, coordinate: Coordinate) -> bool:
        return self._coordinate_data.get_visit_count(coordinate) > 0

    def _clamp_coordinate_to_bounds(self, coordinate: Coordinate) -> Coordinate:

        clamped_coordinate = deepcopy(coordinate)

        for i, v in enumerate(coordinate):
            sd = self._config.get_dimension(i)

            v = min(sd.get_max_idx(), v)
            v = max(sd.get_min_idx(), v)
            clamped_coordinate[i] = v
        return clamped_coordinate

    def _get_covered_values_per_dimension(self) -> List[Dict[Coordinate, bool]]:
        """
        Returns a list of dicts that indicates which values have been
        covered in each dimension.

        (e.g.)
            covered_values_per_dimension[dimension][value] = bool
        """
        visited_coordinates = self._get_visited_coordinates()

        covered_values_per_dimension: List[Dict[Coordinate, bool]] = [
            {} for _ in range(self._config.get_num_dimensions())
        ]

        for coordinate in visited_coordinates:
            for i, v in enumerate(coordinate):
                covered_values_per_dimension[i][v] = True

        return covered_values_per_dimension

    def _get_num_uncovered_values(
            self, coordinate: Coordinate,
            covered_values_per_dimension: List[Dict[Coordinate, bool]]) -> int:
        """
        Determine how many of the coordinate dimensions in the input coordinate have values
        that are not covered in covered_values_per_dimension
        """
        num_uncovered = 0

        for i, v in enumerate(coordinate):
            if not covered_values_per_dimension[i].get(v, False):
                num_uncovered += 1

        return num_uncovered
