# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from model_analyzer.config.generate.neighborhood import Neighborhood
from model_analyzer.config.generate.search_config import SearchConfig
from model_analyzer.config.generate.search_dimension import SearchDimension
from model_analyzer.config.generate.coordinate_data import CoordinateData
from model_analyzer.config.generate.coordinate import Coordinate

from .common import test_result_collector as trc


class TestNeighborhood(trc.TestResultCollector):

    def test_calc_distance(self):
        a = Coordinate([1, 4, 6, 3])
        b = Coordinate([4, 2, 6, 0])

        # Euclidian distance is the square root of the
        # sum of the distances of the coordinates
        #
        # Distance = sqrt( (1-4)^2 + (4-2)^2 + (6-6)^2 + (3-0)^2)
        # Distance = sqrt( 9 + 4 + 0 + 9 )
        # Distance = sqrt(22)
        # Distance = 4.69
        self.assertAlmostEqual(Neighborhood.calc_distance(a, b), 4.69, places=3)

    def test_create_neighborhood(self):
        cd = CoordinateData()

        sc = SearchConfig([
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])
        n = Neighborhood(sc, cd, Coordinate([1, 1, 1]), 2)

        # These are all values within radius of 2 from [1,1,1]
        # but within the bounds (no negative values)
        #
        expected_neighborhood = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0],
                                 [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1],
                                 [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                 [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3],
                                 [1, 2, 0], [1, 2, 1], [1, 2, 2], [1, 3, 1],
                                 [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0],
                                 [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1],
                                 [2, 2, 2], [3, 1, 1]]

        expected_coordinates = [Coordinate(x) for x in expected_neighborhood]
        self.assertEqual(tuple(n._neighborhood), tuple(expected_coordinates))

    def test_num_initialized(self):
        cd = CoordinateData()
        cd.set_throughput(Coordinate([0, 0, 0]), 5)

        sc = SearchConfig([
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])
        n = Neighborhood(sc, cd, Coordinate([1, 1, 1]), 2)

        # Started with 1 initialized
        self.assertEqual(1, n.get_num_initialized_points())

        cd.set_throughput(Coordinate([0, 0, 1]), 5)
        self.assertEqual(2, n.get_num_initialized_points())

        # Set same point. No change to num initialized
        cd.set_throughput(Coordinate([0, 0, 1]), 7)
        self.assertEqual(2, n.get_num_initialized_points())

        # Set a point outside of neighborhood
        cd.set_throughput(Coordinate([0, 0, 4]), 3)
        self.assertEqual(2, n.get_num_initialized_points())

    def test_weighted_center(self):
        sc = SearchConfig([
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])
        cd = CoordinateData()

        n = Neighborhood(sc, cd, Coordinate([1, 1, 1]), 2)

        coordinates = [
            Coordinate([2, 0, 0]),
            Coordinate([0, 1, 0]),
            Coordinate([0, 0, 1])
        ]
        weights = [2, 1, 4]

        weighted_center = n._determine_weighted_coordinate_center(
            coordinates, weights)

        # Weighted center
        #   First multiply each coordinate by its weights:
        #      [4,0,0], [0,1,0], [0,0,4]
        #   Then sum up the coordinates:
        #      [4,1,4]
        #   Then divide by sum of weights (2+1+4=7)
        #      [4/7, 1/7, 4/7]
        expected_weighted_center = Coordinate([4 / 7, 1 / 7, 4 / 7])

        self.assertEqual(weighted_center, expected_weighted_center)

    def test_calculate_new_coordinate_one_dimension(self):
        """ 
        Test calculate_new_coordinate for a case where only
        one of the dimensions increases the throughput
        """
        cd = CoordinateData()
        cd.set_throughput(Coordinate([0, 0]), 2)
        cd.set_throughput(Coordinate([1, 0]), 4)
        cd.set_throughput(Coordinate([0, 1]), 2)

        sc = SearchConfig([
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        n = Neighborhood(sc, cd, Coordinate([0, 0]), 2)

        self.assertEqual(Coordinate([1, 0]), n.calculate_new_coordinate(1))

    def test_calculate_new_coordinate_two_dimensions(self):
        """ 
        Test calculate_new_coordinate for a case where both of the 
        dimensions increases the throughput equally
        """
        cd = CoordinateData()
        cd.set_throughput(Coordinate([0, 0]), 2)
        cd.set_throughput(Coordinate([1, 0]), 4)
        cd.set_throughput(Coordinate([0, 1]), 4)

        sc = SearchConfig([
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        n = Neighborhood(sc, cd, Coordinate([0, 0]), 2)
        magnitude = 1
        self.assertEqual(Coordinate([1, 1]),
                         n.calculate_new_coordinate(magnitude))

    def test_calculate_new_coordinate_larger_magnitude(self):
        """ 
        Test calculate_new_coordinate for a case where both of the 
        dimensions increases the throughput equally, and magnitude is 
        larger than 1
        """
        cd = CoordinateData()
        cd.set_throughput(Coordinate([0, 0]), 2)
        cd.set_throughput(Coordinate([1, 0]), 4)
        cd.set_throughput(Coordinate([0, 1]), 4)

        sc = SearchConfig([
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        n = Neighborhood(sc, cd, Coordinate([0, 0]), 2)
        magnitude = 3

        # Run it multiple times to make sure no values are changing
        self.assertEqual(Coordinate([2, 2]),
                         n.calculate_new_coordinate(magnitude))
        self.assertEqual(Coordinate([2, 2]),
                         n.calculate_new_coordinate(magnitude))
