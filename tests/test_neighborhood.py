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

from unittest.mock import MagicMock

from model_analyzer.config.generate.neighborhood import Neighborhood
from model_analyzer.config.generate.search_config import NeighborhoodConfig
from model_analyzer.config.generate.search_dimension import SearchDimension
from model_analyzer.config.generate.search_dimensions import SearchDimensions
from model_analyzer.config.generate.coordinate_data import CoordinateData
from model_analyzer.config.generate.coordinate import Coordinate

from .common.test_utils import construct_run_config_measurement
from .common import test_result_collector as trc


class TestNeighborhood(trc.TestResultCollector):

    def _construct_rcm(self, throughput: float, latency: float):
        model_name = "modelA"
        model_config_name = ["modelA_config_0"]

        # yapf: disable
        gpu_metric_values = {
            '0': {
                "gpu_used_memory": 6000,
                "gpu_utilization": 60
            },
        }
        non_gpu_metric_values = [{
                "perf_throughput": throughput,
                "perf_latency_avg": latency
        }]
        # yapf: enable

        metric_objectives = [{"perf_throughput": 1}]
        weights = [1]

        rcm = construct_run_config_measurement(
            model_name=model_name,
            model_config_names=model_config_name,
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=gpu_metric_values,
            non_gpu_metric_values=non_gpu_metric_values,
            metric_objectives=metric_objectives,
            model_config_weights=weights
        )
        return rcm

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

        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, cd, home_coordinate=Coordinate([1, 1, 1]))

        rcm = self._construct_rcm(throughput=100, latency=80)

        # Start with 0 initialized
        self.assertEqual(0, n._get_num_initialized_points())
        self.assertFalse(n.enough_coordinates_initialized())

        # Home coordinate is ignored. No change to num initialized.
        cd.set_measurement(Coordinate([1, 1, 1]), rcm)
        self.assertEqual(0, n._get_num_initialized_points())
        self.assertFalse(n.enough_coordinates_initialized())

        # Incremented to 1 initialized
        cd.set_measurement(Coordinate([0, 0, 0]), rcm)
        self.assertEqual(1, n._get_num_initialized_points())
        self.assertFalse(n.enough_coordinates_initialized())

        # Set same point. No change to num initialized
        cd.set_measurement(Coordinate([0, 0, 0]), rcm)
        self.assertEqual(1, n._get_num_initialized_points())
        self.assertFalse(n.enough_coordinates_initialized())

        # Set a point outside of neighborhood
        cd.set_measurement(Coordinate([0, 0, 4]), rcm)
        self.assertEqual(1, n._get_num_initialized_points())
        self.assertFalse(n.enough_coordinates_initialized())

        # Set a third point inside of neighborhood
        cd.set_measurement(Coordinate([1, 0, 0]), rcm)
        self.assertEqual(2, n._get_num_initialized_points())
        self.assertFalse(n.enough_coordinates_initialized())

        # Set the last point inside of neighborhood
        cd.set_measurement(Coordinate([1, 1, 0]), rcm)
        self.assertEqual(3, n._get_num_initialized_points())
        self.assertTrue(n.enough_coordinates_initialized())

    def test_weighted_center(self):
        """
        Test _determine_weighted_center method that computes the target
        weighted center of the coordinates based on their weights.

          1. multiply each coordinates by its weights:
                [2, 0, 0] * 1 = [2, 0, 0]
                [0, 1, 0] * 1.5 = [0, 1.5, 0]
                [0, 0, 1] * 1.5 = [0, 0, 1.5]

          2. sum up the coordinates:
                [2, 1.5, 1.5]

          3. divide by sum of weights (1 + 1.5 + 1.5 = 4.0)
                [1/2, 3/8, 3/8]
        """
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        cd = CoordinateData()
        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, cd, home_coordinate=Coordinate([1, 1, 1]))

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=3, latency=5)
        rcm2 = self._construct_rcm(throughput=7, latency=5)
        rcm3 = self._construct_rcm(throughput=7, latency=5)

        cd.set_measurement(Coordinate([1, 1, 1]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([2, 0, 0]), rcm1)
        cd.set_measurement(Coordinate([0, 1, 0]), rcm2)
        cd.set_measurement(Coordinate([0, 0, 1]), rcm3)

        coordinates, measurements = n._compile_neighborhood_measurements()
        weighted_center = n._determine_weighted_center(
            coordinates=coordinates, measurements=measurements)

        expected_weighted_center = Coordinate([1/2, 3/8, 3/8])
        self.assertEqual(weighted_center, expected_weighted_center)

    def test_calculate_new_coordinate_one_dimension(self):
        """
        Test calculate_new_coordinate for a case where only
        one of the dimensions increases the measurement.
        """
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        cd = CoordinateData()
        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, cd, home_coordinate=Coordinate([0, 0]))

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=3, latency=5)
        rcm2 = self._construct_rcm(throughput=1, latency=5)

        cd.set_measurement(Coordinate([0, 0]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([1, 0]), rcm1)
        cd.set_measurement(Coordinate([0, 1]), rcm2)

        magnitude = 20
        new_coord = n.calculate_new_coordinate(magnitude)
        self.assertEqual(new_coord, Coordinate([20, 0]))

    def test_calculate_new_coordinate_two_dimensions(self):
        """
        Test calculate_new_coordinate for a case where both of the
        dimensions increases the measurement equally.
        """
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        cd = CoordinateData()
        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, cd, home_coordinate=Coordinate([0, 0]))

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=3, latency=5)
        rcm2 = self._construct_rcm(throughput=3, latency=5)

        cd.set_measurement(Coordinate([0, 0]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([1, 0]), rcm1)
        cd.set_measurement(Coordinate([0, 1]), rcm2)

        magnitude = 20
        new_coord = n.calculate_new_coordinate(magnitude)
        self.assertEqual(new_coord, Coordinate([10, 10]))

    def test_calculate_new_coordinate_larger_magnitude(self):
        """
        Test calculate_new_coordinate for a case where both of the
        dimensions increases the measurement equally, and magnitude is
        larger.
        """
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        cd = CoordinateData()
        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, cd, home_coordinate=Coordinate([0, 0]))

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=7, latency=5)
        rcm2 = self._construct_rcm(throughput=7, latency=5)

        cd.set_measurement(Coordinate([0, 0]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([1, 0]), rcm1)
        cd.set_measurement(Coordinate([0, 1]), rcm2)

        magnitude = 30

        # Run it multiple times to make sure no values are changing
        new_coord = n.calculate_new_coordinate(magnitude)
        self.assertEqual(new_coord, Coordinate([15, 15]))
        new_coord = n.calculate_new_coordinate(magnitude)
        self.assertEqual(new_coord, Coordinate([15, 15]))

    def test_calculate_new_coordinate_out_of_bounds(self):
        """
        Test that calculate_new_coordinate will clamp the result to
        the search dimension bounds.

        Both dimensions are defined to only be from 2-7. The test sets up
        the case where the next step WOULD be to [11, -3] if not for bounding
        into the defined range
        """
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension(
                "foo", SearchDimension.DIMENSION_TYPE_LINEAR, min=2, max=7),
            SearchDimension(
                "bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL, min=2, max=7)
        ])

        cd = CoordinateData()
        nc = NeighborhoodConfig(dims, radius=8, min_initialized=3)
        n = Neighborhood(nc, cd, home_coordinate=Coordinate([3, 6]))

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=3, latency=5)

        cd.set_measurement(Coordinate([3, 6]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([4, 5]), rcm1)

        self.assertEqual(Coordinate([7, 2]),
                         n.calculate_new_coordinate(magnitude=8))

    def test_no_magnitude_vector(self):
        """
        Test that if the home coordinate and the weighted_coordinate_center
        are the same, then the step vector is all 0s
        """

        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        cd = CoordinateData()
        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, cd, home_coordinate=Coordinate([1, 1]))

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=3, latency=5)
        rcm2 = self._construct_rcm(throughput=3, latency=5)

        cd.set_measurement(Coordinate([1, 1]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([2, 0]), rcm1)
        cd.set_measurement(Coordinate([0, 2]), rcm2)

        sv = n._get_step_vector()
        expected_sv = Coordinate([0, 0])
        self.assertEqual(sv, expected_sv)

    def test_all_same_throughputs(self):
        """
        Test that when all the coorindates in the neighborhood has the
        same throughputs, the weighted center is same as the home coordinate.
        """
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        cd = CoordinateData()
        nc = NeighborhoodConfig(dims, radius=3, min_initialized=3)
        n = Neighborhood(nc, cd, home_coordinate=Coordinate([0, 0, 0]))

        rcm0 = self._construct_rcm(throughput=10, latency=5)
        rcm1 = self._construct_rcm(throughput=10, latency=5)
        rcm2 = self._construct_rcm(throughput=10, latency=5)
        rcm3 = self._construct_rcm(throughput=10, latency=5)

        cd.set_measurement(Coordinate([0, 0, 0]), rcm0)  # home coordinate
        cd.set_measurement(Coordinate([1, 0, 0]), rcm1)
        cd.set_measurement(Coordinate([0, 1, 0]), rcm2)
        cd.set_measurement(Coordinate([0, 0, 1]), rcm3)

        coordinates, measurements = n._compile_neighborhood_measurements()
        tc = n._determine_weighted_center(coordinates, measurements)
        self.assertEqual(Coordinate([0, 0, 0]), tc)
