# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
from model_analyzer.config.generate.search_config import NeighborhoodConfig
from model_analyzer.config.generate.search_dimension import SearchDimension
from model_analyzer.config.generate.search_dimensions import SearchDimensions
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
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, cd, Coordinate([1, 1, 1]))

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

        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, cd, Coordinate([1, 1, 1]))

        # Started with 1 initialized
        self.assertEqual(1, n._get_num_initialized_points())
        self.assertFalse(n.enough_coordinates_initialized())

        cd.set_throughput(Coordinate([0, 0, 1]), 5)
        self.assertEqual(2, n._get_num_initialized_points())
        self.assertFalse(n.enough_coordinates_initialized())

        # Set same point. No change to num initialized
        cd.set_throughput(Coordinate([0, 0, 1]), 7)
        self.assertEqual(2, n._get_num_initialized_points())

        # Set a point outside of neighborhood
        cd.set_throughput(Coordinate([0, 0, 4]), 3)
        self.assertEqual(2, n._get_num_initialized_points())
        self.assertFalse(n.enough_coordinates_initialized())

        # Set a third point inside of neighborhood
        cd.set_throughput(Coordinate([1, 0, 0]), 9)
        self.assertTrue(n.enough_coordinates_initialized())

    def test_weighted_center(self):
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        cd = CoordinateData()

        n = Neighborhood(nc, cd, Coordinate([1, 1, 1]))

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

        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)

        n = Neighborhood(nc, cd, Coordinate([0, 0]))

        magnitude = 20
        new_coord = n.calculate_new_coordinate(magnitude)

        self.assertEqual(new_coord, Coordinate([3, 0]))

    def test_calculate_new_coordinate_two_dimensions(self):
        """ 
        Test calculate_new_coordinate for a case where both of the 
        dimensions increases the throughput equally
        """
        cd = CoordinateData()
        cd.set_throughput(Coordinate([0, 0]), 2)
        cd.set_throughput(Coordinate([1, 0]), 4)
        cd.set_throughput(Coordinate([0, 1]), 4)

        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)

        n = Neighborhood(nc, cd, Coordinate([0, 0]))
        magnitude = 20
        new_coord = n.calculate_new_coordinate(magnitude)

        self.assertEqual(new_coord, Coordinate([1, 1]))

    def test_calculate_new_coordinate_larger_magnitude(self):
        """ 
        Test calculate_new_coordinate for a case where both of the 
        dimensions increases the throughput equally, and magnitude is 
        larger than 1
        """
        cd = CoordinateData()
        cd.set_throughput(Coordinate([0, 0]), 2)
        cd.set_throughput(Coordinate([1, 0]), 8)
        cd.set_throughput(Coordinate([0, 1]), 8)

        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)

        n = Neighborhood(nc, cd, Coordinate([0, 0]))
        magnitude = 20
        new_coord = n.calculate_new_coordinate(magnitude)

        # Run it multiple times to make sure no values are changing
        self.assertEqual(new_coord, Coordinate([2, 2]))
        self.assertEqual(new_coord, Coordinate([2, 2]))

    def test_calculate_new_coordinate_out_of_bounds(self):
        """ 
        Test that calculate_new_coordinate will clamp the result to
        the search dimention bounds

        Both dimensions are defined to only be from 2-7. The test sets up 
        the case where the next step WOULD be to [1,8] if not for bounding
        into the defined range
        """
        cd = CoordinateData()
        cd.set_throughput(Coordinate([3, 6]), 100)
        cd.set_throughput(Coordinate([4, 5]), 1)

        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension(
                "foo", SearchDimension.DIMENSION_TYPE_LINEAR, min=2, max=7),
            SearchDimension(
                "bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL, min=2, max=7)
        ])

        nc = NeighborhoodConfig(dims, radius=8, min_initialized=3)

        n = Neighborhood(nc, cd, Coordinate([3, 6]))

        self.assertEqual(Coordinate([2, 7]),
                         n.calculate_new_coordinate(magnitude=3))

    def test_no_magnitude_unit_vector(self):
        """
        Test that if the coordinate_center and weighted_coordinate_center
        are the same, then the step vector is all 0s
        """
        cd = CoordinateData()
        cd.set_throughput(Coordinate([1, 0]), 4)
        cd.set_throughput(Coordinate([0, 1]), 4)

        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)

        n = Neighborhood(nc, cd, Coordinate([0, 0]))

        uv = n._get_step_vector()
        expected_uv = Coordinate([0, 0])
        self.assertEqual(uv, expected_uv)
