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

        self._force_slow_mode = False

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

        If the neighborhood is in slow mode, this means all adjacent neighbors
        must be visited
        """
        if self._is_slow_mode():
            return self._are_all_adjacent_neighbors_visited()
        else:
            min_initialized = self._config.get_min_initialized()
            num_initialized = len(self._get_initialized_coordinates())
            return num_initialized >= min_initialized

    def force_slow_mode(self):
        """
        When called, forces the neighborhood into slow mode
        """
        self._force_slow_mode = True

    def calculate_new_coordinate(self,
                                 magnitude: int,
                                 enable_clipping: bool = True,
                                 clip_value: int = 2) -> Coordinate:
        """
        Based on the measurements in the neighborhood, determine where
        the next location should be.

        If the neighborhood is in slow mode, return the best found measurement
        Otherwise calculate a new coordinate from the measurements

        Parameters
        ----------
        magnitude
            How large of a step to take
        enable_clipping
            Determines whether or not to clip the final step vector.
        clip_value
            What value to clip the vector at, if it is enabled

        Returns
        -------
        new_coordinate
            The new coordinate computed based on the neighborhood measurements.
        """

        if self._is_slow_mode():
            return self._get_best_coordinate_found()
        else:
            return self._calculate_new_coordinate(magnitude, enable_clipping,
                                                  clip_value)

    def _get_best_coordinate_found(self) -> Coordinate:
        vectors, measurements = self._get_measurements_passing_constraints()

        if len(vectors) == 0:
            return self._home_coordinate

        home_measurement = self._get_home_measurement()

        if home_measurement.is_passing_constraints():
            vectors.append(Coordinate([0] * self._config.get_num_dimensions()))
            measurements.append(home_measurement)

        best_vector = vectors[0]
        best_measurement = measurements[0]

        for v, m in zip(vectors, measurements):
            if m > best_measurement:
                best_vector = v
                best_measurement = m

        best_coordinate = self._home_coordinate + best_vector
        return best_coordinate

    def _calculate_new_coordinate(self, magnitude, enable_clipping,
                                  clip_value) -> Coordinate:

        step_vector = self._get_step_vector() * magnitude

        if enable_clipping:
            step_vector = self._clip_vector_values(vector=step_vector,
                                                   clip_value=clip_value)
        step_vector.round()

        tmp_new_coordinate = self._home_coordinate + step_vector
        new_coordinate = self._clamp_coordinate_to_bounds(tmp_new_coordinate)
        return new_coordinate

    def _clip_vector_values(self, vector: Coordinate,
                            clip_value: int) -> Coordinate:
        """
        Clip the values of the vector to be within the range of
        [-clip_value, clip_value]. The clipping preserves the direction
        of the vector (e.g. if the clip_value is 2, then [10, 5] will be
        clipped to [2, 1] instead of [2, 2]).

        Parameters
        ----------
        vector
            an input vector that may require clipping
        clip_value
            a non-negative integer that bounds the values of the input vector

        Returns
        -------
        vector
            a vector with all of its values within [-clip_value, clip_value]
        """
        assert clip_value >= 0, "clip_value must be non-negative number."

        max_value = max(abs(c) for c in vector)

        if max_value > clip_value and max_value != 0:
            for i in range(len(vector)):
                vector[i] = clip_value * vector[i] / max_value
        return vector

    def pick_coordinate_to_initialize(self) -> Optional[Coordinate]:
        """
        Based on the initialized coordinate values, pick an unvisited
        coordinate to initialize next.

        If the neighborhood is in slow mode, only pick from within the adjacent neighbors
        """

        if self._is_slow_mode():
            return self._pick_slow_mode_coordinate_to_initialize()
        else:
            return self._pick_fast_mode_coordinate_to_initialize()

    def _pick_slow_mode_coordinate_to_initialize(self):
        for neighbor in self._get_all_adjacent_neighbors():
            if not self._is_coordinate_visited(neighbor):
                return neighbor

        raise Exception("Picking slow mode coordinate, but none are unvisited")

    def _pick_fast_mode_coordinate_to_initialize(self):
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

    def _get_initialized_coordinates(self) -> List[Coordinate]:
        initialized_coordinates = []
        for coordinate in self._neighborhood:
            if coordinate != self._home_coordinate \
                    and self._is_coordinate_initialized(coordinate):
                initialized_coordinates.append(deepcopy(coordinate))
        return initialized_coordinates

    def _get_step_vector(self) -> Coordinate:
        """
        Calculate a vector that indicates a direction to step from the
        home coordinate (current center).

        If the home coordinate is passing the constraints, the method
        calculates the step vector based on the objectives given.
        Otherwise, it calculates the step vector based on the constraints
        so that it steps toward the region that passes the constraints.

        Returns
        -------
        step_vector
            a coordinate that tells the direction to move.
        """
        home_measurement = self._get_home_measurement()

        assert home_measurement is not None, "Home measurement cannot be NoneType."

        if self._is_home_passing_constraints():
            return self._optimize_for_objectives(home_measurement)

        return self._optimize_for_constraints(home_measurement)

    def _optimize_for_objectives(
            self, home_measurement: RunConfigMeasurement) -> Coordinate:
        """
        Calculate a step vector that maximizes the current objectives.
        If no vectors are provided, return zero vector.

        Parameters
        ----------
        vectors
            list of vectors from home coordinate to the neighboring
            coordinates that are passing constraints
        measurements
            list of measurements of the neighboring coordinates that
            are passing constraints

        Returns
        -------
        step_vector
            a coordinate that tells the direction to move.
        """
        vectors, measurements = self._get_measurements_passing_constraints()
        step_vector = Coordinate([0] * self._config.get_num_dimensions())

        if not vectors:
            return step_vector

        for vector, measurement in zip(vectors, measurements):
            weight = home_measurement.compare_measurements(measurement)
            step_vector += vector * weight

        step_vector /= len(vectors)
        return step_vector

    def _optimize_for_constraints(
            self, home_measurement: RunConfigMeasurement) -> Coordinate:
        """
        Calculate a step vector that steps toward the coordinates that
        pass the constraints (set default weights to 1.0). When no vectors
        are provided (meaning there are no neighbors passing constraints),
        continue with the neighbors that are not passing constraints by
        comparing how much they are close to passing constraints.

        Parameters
        ----------
        vectors
            list of vectors from home coordinate to the neighboring
            coordinates that are passing constraints
        measurements
            list of measurements of the neighboring coordinates that
            are passing constraints

        Returns
        -------
        step_vector
            a coordinate that tells the direction to move.
        """
        vectors, measurements = self._get_measurements_passing_constraints()
        step_vector = Coordinate([0] * self._config.get_num_dimensions())

        if not vectors:
            vectors, measurements = self._get_all_visited_measurements()

        for vector, measurement in zip(vectors, measurements):
            weight = home_measurement.compare_constraints(measurement)
            if measurement.is_passing_constraints():
                weight = 1.0  # when home fails & neighbor passes

            step_vector += vector * weight

        step_vector /= len(vectors)
        return step_vector

    def _get_all_visited_measurements(
            self) -> Tuple[List[Coordinate], List[RunConfigMeasurement]]:
        """
        Gather all the visited vectors (directions from the home coordinate)
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
            if measurement:
                vectors.append(coordinate - self._home_coordinate)
                measurements.append(measurement)
        return vectors, measurements

    def _get_measurements_passing_constraints(
            self) -> Tuple[List[Coordinate], List[RunConfigMeasurement]]:
        """
        Gather all the vectors (directions from the home coordinate)
        and their corresponding measurements that are passing constraints.

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
            if measurement and measurement.is_passing_constraints():
                vectors.append(coordinate - self._home_coordinate)
                measurements.append(measurement)
        return vectors, measurements

    def _is_coordinate_visited(self, coordinate: Coordinate) -> bool:
        return self._coordinate_data.get_visit_count(coordinate) > 0

    def _is_coordinate_initialized(self, coordinate: Coordinate) -> bool:
        return self._coordinate_data.get_measurement(coordinate) is not None

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

    def _is_slow_mode(self):
        """ 
        Returns true if any of the following are true:
        - force_slow_mode() has been called
        - Home is passing and anyone in the neighborhood is failing
        - Home is failing and anyone in the neighborhood is passing
        """
        if self._force_slow_mode:
            return True

        if not self._is_home_measured():
            return False

        passing_vectors, _ = self._get_measurements_passing_constraints()
        all_vectors, _ = self._get_all_visited_measurements()

        any_failing = len(all_vectors) != len(passing_vectors)
        any_passing = len(passing_vectors) != 0
        home_passing = self._is_home_passing_constraints()

        return (home_passing and any_failing) or (not home_passing and
                                                  any_passing)

    def _are_all_adjacent_neighbors_visited(self):
        for neighbor in self._get_all_adjacent_neighbors():
            if not self._is_coordinate_visited(neighbor):
                return False
        return True

    def _get_all_adjacent_neighbors(self):
        adjacent_neighbors = []

        for dim in range(self._config.get_num_dimensions()):
            dimension = self._config.get_dimension(dim)

            down_neighbor = Coordinate(self._home_coordinate)
            down_neighbor[dim] -= 1
            if down_neighbor[dim] >= dimension.get_min_idx():
                adjacent_neighbors.append(down_neighbor)

            up_neighbor = Coordinate(self._home_coordinate)
            up_neighbor[dim] += 1
            if up_neighbor[dim] <= dimension.get_max_idx():
                adjacent_neighbors.append(up_neighbor)

        return adjacent_neighbors

    def _get_home_measurement(self):
        return self._coordinate_data.get_measurement(
            coordinate=self._home_coordinate)

    def _is_home_measured(self):
        return self._get_home_measurement() is not None

    def _is_home_passing_constraints(self):
        if not self._is_home_measured():
            raise Exception("Can't check home passing if it isn't measured yet")

        home_measurement = self._get_home_measurement()
        return home_measurement.is_passing_constraints()
