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

    # This defines the bounds of how the vector calculated from
    # measurements is converted to a step vector.
    #
    # The translation will return the lowest index that has a value greater
    # than the input value.
    #
    # For example, if the input is greater than the value in index 1 but less than
    # the value in index 2, the resulting step will be 1
    #
    TRANSLATION_LIST = [0.09, 0.3, 1.0]

    def __init__(self, neighborhood_config: NeighborhoodConfig,
                 home_coordinate: Coordinate, coordinate_data: CoordinateData):
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
        self._coordinate_data = coordinate_data

        self._radius = self._config.get_radius()
        self._neighborhood = self._create_neighborhood()

        self._force_slow_mode = False

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
        have been initialized with valid measurements. Else false

        If the neighborhood is in slow mode, this means all adjacent neighbors
        must be visited
        """
        if self._is_slow_mode():
            return self._are_all_adjacent_neighbors_measured()
        else:
            min_initialized = self._config.get_min_initialized()
            num_initialized = len(
                self._get_coordinates_with_valid_measurements())
            return num_initialized >= min_initialized

    def force_slow_mode(self) -> None:
        """
        When called, forces the neighborhood into slow mode
        """
        self._force_slow_mode = True

    def determine_new_home(self) -> Coordinate:
        """
        Based on the measurements in the neighborhood, determine where
        the next location should be.

        If the neighborhood is in slow mode, return the best found measurement
        Otherwise calculate a new coordinate from the measurements

        Returns
        -------
        new_coordinate
            The new coordinate computed based on the neighborhood measurements.
        """

        if self._is_slow_mode():
            return self._get_best_coordinate_found()
        else:
            return self._calculate_new_home()

    def _get_best_coordinate_found(self) -> Coordinate:
        vectors, measurements = self._get_measurements_passing_constraints()

        if len(vectors) == 0:
            return self._home_coordinate

        home_measurement = self._get_home_measurement()

        if home_measurement and home_measurement.is_passing_constraints():
            vectors.append(Coordinate([0] * self._config.get_num_dimensions()))
            measurements.append(home_measurement)

        _, best_vector = sorted(zip(measurements, vectors))[-1]

        best_coordinate = self._home_coordinate + best_vector
        return best_coordinate

    def _calculate_new_home(self) -> Coordinate:
        step_vector = self._get_step_vector()
        step_vector_coordinate = self._translate_step_vector(
            step_vector, Neighborhood.TRANSLATION_LIST)
        tmp_new_coordinate = self._home_coordinate + step_vector_coordinate
        new_coordinate = self._clamp_coordinate_to_bounds(tmp_new_coordinate)
        return new_coordinate

    def _translate_step_vector(self, step_vector: List[float],
                               translate_list: List[float]) -> Coordinate:

        translated_step_vector = Coordinate([0] * len(step_vector))
        for i, v in enumerate(step_vector):
            translated_step_vector[i] = self._translate_value(v, translate_list)

        return translated_step_vector

    def _translate_value(self, value: float,
                         translation_list: List[float]) -> int:
        ret = 0
        for index, bound in enumerate(translation_list):
            if value > 0 and value > bound:
                ret = index + 1
            if value < 0 and value < -1 * bound:
                ret = -1 * (index + 1)
        return ret

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

    def _pick_slow_mode_coordinate_to_initialize(self) -> Coordinate:
        for neighbor in self._get_all_adjacent_neighbors():
            if not self._is_coordinate_measured(neighbor):
                return neighbor

        raise Exception("Picking slow mode coordinate, but none are unvisited")

    def _pick_fast_mode_coordinate_to_initialize(self) -> Optional[Coordinate]:
        covered_values_per_dimension = self._get_covered_values_per_dimension()

        max_num_uncovered = -1
        best_coordinate = None
        for coordinate in self._neighborhood:
            if not self._is_coordinate_measured(coordinate):
                num_uncovered = self._get_num_uncovered_values(
                    coordinate, covered_values_per_dimension)

                if num_uncovered > max_num_uncovered:
                    max_num_uncovered = num_uncovered
                    best_coordinate = coordinate

        return best_coordinate

    def get_nearest_neighbor(self, coordinate_in: Coordinate) -> Coordinate:
        """
        Find the nearest coordinate to the `coordinate_in` among the
        coordinates within the current neighborhood.
        """
        min_distance = float('inf')
        nearest_neighbor = self._home_coordinate

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

            lower_bound = max(dimension.get_min_idx(),
                              int(coordinate[i]) - radius)
            upper_bound = min(dimension.get_max_idx(),
                              int(coordinate[i]) + radius + 1)
            bounds.append([lower_bound, upper_bound])
        return bounds

    def _enumerate_all_values_in_bounds(
            self, bounds: List[List[int]]) -> List[List[int]]:
        possible_index_values = []
        for bound in bounds:
            low: int = bound[0]
            high: int = bound[1]
            possible_index_values.append(list(range(low, high + 1)))
        tuples = list(product(*possible_index_values))
        return [list(x) for x in tuples]

    def _get_coordinates_with_valid_measurements(self) -> List[Coordinate]:
        initialized_coordinates = []
        for coordinate in self._neighborhood:
            if coordinate != self._home_coordinate and self._coordinate_data.has_valid_measurement(
                    coordinate):
                initialized_coordinates.append(deepcopy(coordinate))
        return initialized_coordinates

    def _get_step_vector(self) -> List[float]:
        """
        Calculate a vector that indicates a direction to step from the
        home coordinate (current center).

        Returns
        -------
        step_vector
            a coordinate that tells the direction to move.
        """

        compare_constraints = not self._is_home_passing_constraints()
        return self._calculate_step_vector_from_measurements(
            compare_constraints=compare_constraints)

    def _calculate_step_vector_from_measurements(
            self, compare_constraints: bool) -> List[float]:

        home_measurement = self._get_home_measurement()
        if not home_measurement:
            raise Exception("Can't step from home if it has no measurement")

        vectors, measurements = self._get_all_measurements()

        # This function should only ever be called if all are passing or none are passing
        _, p = self._get_measurements_passing_constraints()
        assert (len(p) == 0 or len(p) == len(measurements))

        if not vectors:
            return ([0.0] * self._config.get_num_dimensions())

        weights = []
        for m in measurements:
            if compare_constraints:
                weight = home_measurement.compare_constraints(m)
            else:
                weight = home_measurement.compare_measurements(m)
            if not weight:
                weight = 0.0
            weights.append(weight)

        return self._calculate_step_vector_from_vectors_and_weights(
            vectors, weights)

    def _calculate_step_vector_from_vectors_and_weights(
            self, vectors: List[Coordinate],
            weights: List[float]) -> List[float]:
        step_vector = [0.0] * self._config.get_num_dimensions()
        dim_sum_vector = [0.0] * self._config.get_num_dimensions()

        # For each dimension -
        #   if non zero, add weight (inverting if dimension is negative)
        #   divide by sum of coordinate of that dimension
        for vector, weight in zip(vectors, weights):

            for dim, v in enumerate(vector):
                if v:
                    if v > 0:
                        step_vector[dim] += weight
                        dim_sum_vector[dim] += v
                    else:
                        step_vector[dim] -= weight
                        dim_sum_vector[dim] -= v

        for dim, v in enumerate(dim_sum_vector):
            if v:
                step_vector[dim] /= v

        return step_vector

    def _get_all_measurements(
            self) -> Tuple[List[Coordinate], List[RunConfigMeasurement]]:
        """
        Gather all the visited vectors (directions from the home coordinate)
        and their corresponding measurements.

        Returns
        -------
        (vectors, measurements)
            collection of vectors and their measurements.
        """
        coordinates = self._get_coordinates_with_valid_measurements()

        vectors = []
        measurements = []
        for coordinate in coordinates:
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
        coordinates = self._get_coordinates_with_valid_measurements()

        vectors = []
        measurements = []
        for coordinate in coordinates:
            measurement = self._coordinate_data.get_measurement(coordinate)
            if measurement and measurement.is_passing_constraints():
                vectors.append(coordinate - self._home_coordinate)
                measurements.append(measurement)
        return vectors, measurements

    def _is_coordinate_measured(self, coordinate: Coordinate) -> bool:
        return self._coordinate_data.is_measured(coordinate)

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
        measured_coordinates = self._get_coordinates_with_valid_measurements()

        covered_values_per_dimension: List[Dict[Coordinate, bool]] = [
            {} for _ in range(self._config.get_num_dimensions())
        ]

        for coordinate in measured_coordinates:
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

    def _is_slow_mode(self) -> bool:
        if self._force_slow_mode:
            return True

        if not self._is_home_measured():
            return False

        passing_vectors, _ = self._get_measurements_passing_constraints()
        all_vectors, _ = self._get_all_measurements()

        any_failing = len(all_vectors) != len(passing_vectors)
        any_passing = len(passing_vectors) != 0
        home_passing = self._is_home_passing_constraints()

        return (home_passing and any_failing) or (not home_passing and
                                                  any_passing)

    def _are_all_adjacent_neighbors_measured(self) -> bool:
        for neighbor in self._get_all_adjacent_neighbors():
            if not self._is_coordinate_measured(neighbor):
                return False
        return True

    def _get_all_adjacent_neighbors(self) -> List[Coordinate]:
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

    def _get_home_measurement(self) -> Optional[RunConfigMeasurement]:
        return self._coordinate_data.get_measurement(
            coordinate=self._home_coordinate)

    def _is_home_measured(self) -> bool:
        return self._get_home_measurement() is not None

    def _is_home_passing_constraints(self) -> bool:
        home_measurement = self._get_home_measurement()
        if not home_measurement:
            raise Exception("Can't check home passing if it isn't measured yet")

        return home_measurement.is_passing_constraints()
