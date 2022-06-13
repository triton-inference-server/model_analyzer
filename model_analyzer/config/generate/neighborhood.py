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

from model_analyzer.config.generate.coordinate import Coordinate
from model_analyzer.config.generate.search_config import NeighborhoodConfig


class Neighborhood:
    """
    Defines and operates on a set of coordinates within a radius around
    a 'home' coordinate
    """

    def __init__(self, neighborhood_config, coordinate_data, home_coordinate):
        """
        Parameters
        ----------
        neighborhood_config: 
            NeighborhoodConfig object
        coordinate_data: 
            CoordinateData object
        home_coordinate: 
            Coordinate object to center the neighborhood around
        """
        assert type(neighborhood_config) == NeighborhoodConfig

        self._config = neighborhood_config
        self._coordinate_data = coordinate_data
        self._home_coordinate = home_coordinate

        self._radius = self._config.get_radius()
        self._neighborhood = self._create_neighborhood()

    @classmethod
    def calc_distance(cls, coordinate1, coordinate2):
        """ 
        Return the euclidean distance between two coordinates
        """

        distance = 0
        for i, _ in enumerate(coordinate1):
            diff = coordinate1[i] - coordinate2[i]
            distance += math.pow(diff, 2)
        distance = math.sqrt(distance)
        return distance

    def enough_coordinates_initialized(self):
        """
        Returns true if enough coordinates inside of the neighborhood
        have been initialized. Else false
        """
        min_initialized = self._config.get_min_initialized()
        num_initialized = self._get_num_initialized_points()
        return num_initialized >= min_initialized

    def calculate_new_coordinate(self, magnitude):
        """
        Based on the throughputs in the neighborhood, determine where
        the next location should be

        magnitude: int
            How large of a step to take

        returns: Coordinate
        """
        unit_vector = self._get_unit_vector()
        tmp_new_coordinate = self._home_coordinate + round(
            unit_vector * magnitude)
        new_coordinate = self._clamp_coordinate_to_bounds(tmp_new_coordinate)

        return new_coordinate

    def pick_coordinate_to_initialize(self):
        """
        Based on the initialized coordinate values, pick an uninitialized coordinate to initialize
        """
        covered_values_per_dimension = self._get_covered_values_per_dimension()

        max_num_uncovered = -1
        best_coordinate = None
        for coordinate in self._neighborhood:
            if not self._is_coordinate_initialized(coordinate):
                num_uncovered = self._get_num_uncovered_values(
                    coordinate, covered_values_per_dimension)

                if num_uncovered > max_num_uncovered:
                    max_num_uncovered = num_uncovered
                    best_coordinate = coordinate

        return best_coordinate

    def get_nearest_unvisited_neighbor(self, coordinate_in):
        """ Returns the nearest unvisited coordinate to coordinate_in """
        min_distance = None
        nearest_uninitialized_neighbor = None

        for coordinate in self._neighborhood:

            if self._is_coordinate_visited(coordinate):
                continue

            distance = Neighborhood.calc_distance(coordinate, coordinate_in)
            if not min_distance or distance < min_distance:
                nearest_uninitialized_neighbor = coordinate
                min_distance = distance

        return nearest_uninitialized_neighbor

    def _create_neighborhood(self):

        neighborhood = []
        potential_neighborhood = self._get_potential_neighborhood(
            self._home_coordinate, self._radius)

        for potential_coordinate in potential_neighborhood:
            distance = Neighborhood.calc_distance(self._home_coordinate,
                                                  potential_coordinate)

            if distance <= self._radius:
                neighborhood.append(potential_coordinate)

        return neighborhood

    def _get_potential_neighborhood(self, coordinate, radius):
        bounds = self._get_bounds(coordinate, radius)
        potential_values = self._enumerate_all_values_in_bounds(bounds)
        return [Coordinate(x) for x in potential_values]

    def _get_bounds(self, coordinate, radius):
        bounds = []
        for i in range(self._config.get_num_dimensions()):
            dimension = self._config.get_dimension(i)

            lower_bound = max(dimension.get_min_idx(), coordinate[i] - radius)
            upper_bound = min(dimension.get_max_idx(),
                              coordinate[i] + radius + 1)
            bounds.append([lower_bound, upper_bound])
        return bounds

    def _enumerate_all_values_in_bounds(self, bounds):
        possible_index_values = []
        for bound in bounds:
            possible_index_values.append(list(range(bound[0], bound[1])))

        tuples = list(product(*possible_index_values))
        return [list(x) for x in tuples]

    def _get_num_initialized_points(self):
        """ 
        Returns the number of coordinates in the neighborhood that have a throughput
        associated with it
        """
        num_initialized = 0
        for coordinate in self._neighborhood:
            if self._is_coordinate_initialized(coordinate):
                num_initialized += 1
        return num_initialized

    def _get_unit_vector(self):
        """
        Create a unit vector pointing from the unweighted center of the
        datapoints inside of the neighborhood to the weighted center of 
        the datapoints inside of the neighborhood
        """
        coordinates, throughputs = self._compile_neighborhood_throughputs()

        coordinate_center = self._determine_coordinate_center(coordinates)
        throughput_center = self._determine_weighted_coordinate_center(
            coordinates, throughputs)

        vector = throughput_center - coordinate_center

        unit_vector = self._convert_to_unit_vector(vector)
        return unit_vector

    def _compile_neighborhood_throughputs(self):
        coordinates = []
        throughputs = []
        for coordinate in self._neighborhood:
            throughput = self._coordinate_data.get_throughput(coordinate)
            if throughput is not None:
                coordinates.append(deepcopy(coordinate))
                throughputs.append(throughput)
        return coordinates, throughputs

    def _determine_coordinate_center(self, coordinates):
        coordinate_center = Coordinate([0] * self._config.get_num_dimensions())

        for coordinate in coordinates:
            coordinate_center += coordinate

        num_coordinates = len(coordinates)
        coordinate_center /= num_coordinates

        return coordinate_center

    def _determine_weighted_coordinate_center(self, coordinates, weights):
        weighted_center = Coordinate([0] * self._config.get_num_dimensions())

        for i, _ in enumerate(weights):
            weighted_center += coordinates[i] * weights[i]

        weights_sum = sum(weights)

        weighted_center /= weights_sum

        return weighted_center

    def _convert_to_unit_vector(self, vector):
        magnitude = 0
        for v in vector:
            magnitude += math.pow(v, 2)
        magnitude = math.sqrt(magnitude)

        # Convert the vector to unit vector
        if magnitude == 0:
            unit_vector = vector
        else:
            unit_vector = vector / magnitude

        return unit_vector

    def _is_coordinate_initialized(self, coordinate):
        return self._coordinate_data.get_throughput(coordinate) is not None

    def _is_coordinate_visited(self, coordinate):
        return self._coordinate_data.get_visit_count(coordinate) > 0

    def _clamp_coordinate_to_bounds(self, coordinate):

        clamped_coordiante = deepcopy(coordinate)

        for i, v in enumerate(coordinate):
            sd = self._config.get_dimension(i)

            v = min(sd.get_max_idx(), v)
            v = max(sd.get_min_idx(), v)
            clamped_coordiante[i] = v
        return clamped_coordiante

    def _get_covered_values_per_dimension(self):
        """
        Returns list of lists indicating which values have been covered in each dimension

            covered_values_per_dimension[dimension][value] = bool

        """
        initialized_coordinates, _ = self._compile_neighborhood_throughputs()

        covered_values_per_dimension = [
            {} for _ in range(self._config.get_num_dimensions())
        ]

        for coordinate in initialized_coordinates:
            for i, v in enumerate(coordinate):
                covered_values_per_dimension[i][v] = True

        return covered_values_per_dimension

    def _get_num_uncovered_values(self, coordinate,
                                  covered_values_per_dimension):
        """
        Determine how many of the coordinate dimensions in the input coordinate have values
        that are not covered in covered_values_per_dimension
        """
        num_uncovered = 0

        for i, v in enumerate(coordinate):
            if not covered_values_per_dimension[i].get(v, False):
                num_uncovered += 1

        return num_uncovered
