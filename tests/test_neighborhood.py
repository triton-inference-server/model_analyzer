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
            gpu_metric_values={},
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
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, home_coordinate=Coordinate([1, 1, 1]))

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
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, home_coordinate=Coordinate([1, 1, 1]))

        rcm = self._construct_rcm(throughput=100, latency=80)

        # Start with 0 initialized
        self.assertEqual(0, len(n._get_visited_coordinates()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Home coordinate is ignored. No change to num initialized/visited.
        n.coordinate_data.set_measurement(Coordinate([1, 1, 1]), rcm)
        n.coordinate_data.increment_visit_count(Coordinate([1, 1, 1]))
        self.assertEqual(0, len(n._get_visited_coordinates()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Incremented to 1 initialized
        n.coordinate_data.set_measurement(Coordinate([0, 0, 0]), rcm)
        n.coordinate_data.increment_visit_count(Coordinate([0, 0, 0]))
        self.assertEqual(1, len(n._get_visited_coordinates()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Set same point. No change to num initialized
        n.coordinate_data.set_measurement(Coordinate([0, 0, 0]), rcm)
        n.coordinate_data.increment_visit_count(Coordinate([0, 0, 0]))
        self.assertEqual(1, len(n._get_visited_coordinates()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Set a point outside of neighborhood
        n.coordinate_data.set_measurement(Coordinate([0, 0, 4]), rcm)
        n.coordinate_data.increment_visit_count(Coordinate([0, 0, 4]))
        self.assertEqual(1, len(n._get_visited_coordinates()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Set a third point inside of neighborhood
        n.coordinate_data.set_measurement(Coordinate([1, 0, 0]), rcm)
        n.coordinate_data.increment_visit_count(Coordinate([1, 0, 0]))
        self.assertEqual(2, len(n._get_visited_coordinates()))
        self.assertFalse(n.enough_coordinates_initialized())

        # Set the last point inside of neighborhood
        n.coordinate_data.set_measurement(Coordinate([1, 1, 0]), rcm)
        n.coordinate_data.increment_visit_count(Coordinate([1, 1, 0]))
        self.assertEqual(3, len(n._get_visited_coordinates()))
        self.assertTrue(n.enough_coordinates_initialized())

    def test_step_vector(self):
        """
        Test _get_step_vector method that determines the direction to step
        towards from the home coordinate using the collected coordinates
        and their measurements.

          1. Get vectors from home to the candidate coordinates
             and their measurements.
                [2, 1, 1] - [1, 1, 1] = [1, 0, 0]
                [1, 2, 1] - [1, 1, 1] = [0, 1, 0]

          2. Multiply each vectors by its weights:
                [1, 0, 0] * 1.0 = [1, 0, 0]
                [0, 1, 0] * 1.0 = [0, 1, 0]

          2. Compute the average of the vectors:
                ([1, 0, 0] + [0, 1, 0]) / 2 = [1/2, 1/2, 0]
        """
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, home_coordinate=Coordinate([1, 1, 1]))

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=3, latency=5)
        rcm2 = self._construct_rcm(throughput=3, latency=5)

        n.coordinate_data.set_measurement(Coordinate([1, 1, 1]), rcm0)  # home coordinate
        n.coordinate_data.increment_visit_count(Coordinate([1, 1, 1]))

        n.coordinate_data.set_measurement(Coordinate([2, 1, 1]), rcm1)
        n.coordinate_data.increment_visit_count(Coordinate([2, 1, 1]))

        n.coordinate_data.set_measurement(Coordinate([1, 2, 1]), rcm2)
        n.coordinate_data.increment_visit_count(Coordinate([1, 2, 1]))

        step_vector = n._get_step_vector()
        expected_step_vector = Coordinate([1/2, 1/2, 0])
        self.assertEqual(step_vector, expected_step_vector)

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

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, home_coordinate=Coordinate([0, 0]))

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=3, latency=5)
        rcm2 = self._construct_rcm(throughput=1, latency=5)

        n.coordinate_data.set_measurement(Coordinate([0, 0]), rcm0)  # home coordinate
        n.coordinate_data.increment_visit_count(Coordinate([0, 0]))

        n.coordinate_data.set_measurement(Coordinate([1, 0]), rcm1)
        n.coordinate_data.increment_visit_count(Coordinate([1, 0]))

        n.coordinate_data.set_measurement(Coordinate([0, 1]), rcm2)
        n.coordinate_data.increment_visit_count(Coordinate([0, 1]))

        magnitude = 20
        new_coord = n.calculate_new_coordinate(magnitude,
                                               disable_clipping=True)
        self.assertEqual(new_coord, Coordinate([10, 0]))

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

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, home_coordinate=Coordinate([0, 0]))

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=3, latency=5)
        rcm2 = self._construct_rcm(throughput=3, latency=5)

        n.coordinate_data.set_measurement(Coordinate([0, 0]), rcm0)  # home coordinate
        n.coordinate_data.increment_visit_count(Coordinate([0, 0]))

        n.coordinate_data.set_measurement(Coordinate([1, 0]), rcm1)
        n.coordinate_data.increment_visit_count(Coordinate([1, 0]))

        n.coordinate_data.set_measurement(Coordinate([0, 1]), rcm2)
        n.coordinate_data.increment_visit_count(Coordinate([0, 1]))

        magnitude = 20
        new_coord = n.calculate_new_coordinate(magnitude,
                                               disable_clipping=True)
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

        nc = NeighborhoodConfig(dims, radius=2, min_initialized=3)
        n = Neighborhood(nc, home_coordinate=Coordinate([0, 0]))

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=7, latency=5)
        rcm2 = self._construct_rcm(throughput=7, latency=5)

        n.coordinate_data.set_measurement(Coordinate([0, 0]), rcm0)  # home coordinate
        n.coordinate_data.increment_visit_count(Coordinate([0, 0]))

        n.coordinate_data.set_measurement(Coordinate([1, 0]), rcm1)
        n.coordinate_data.increment_visit_count(Coordinate([1, 0]))

        n.coordinate_data.set_measurement(Coordinate([0, 1]), rcm2)
        n.coordinate_data.increment_visit_count(Coordinate([0, 1]))

        magnitude = 30

        # Run it multiple times to make sure no values are changing
        new_coord = n.calculate_new_coordinate(magnitude,
                                               disable_clipping=True)
        self.assertEqual(new_coord, Coordinate([22, 22]))

        new_coord = n.calculate_new_coordinate(magnitude,
                                               disable_clipping=True)
        self.assertEqual(new_coord, Coordinate([22, 22]))

    def test_calculate_new_coordinate_out_of_bounds(self):
        """
        Test that calculate_new_coordinate will clamp the result to
        the search dimension bounds.

        Both dimensions are defined to only be from 2-7. The test sets up
        the case where the next step WOULD be to [11, -2] if not for bounding
        into the defined range.

          1. Compute the candidate vectors:
              [4, 5] - [3, 6] = [1, -1]

          2. Compute the step vector:
              8 * [1, -1] = [8, -8]

          3. Calculate new coordinate:
              [3, 6] + [8, -8] = [11, -2]

          4. Clamp the new coordinate within bounds:
              [11, -2] -> [7, 2]
        """
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension(
                "foo", SearchDimension.DIMENSION_TYPE_LINEAR, min=2, max=7),
            SearchDimension(
                "bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL, min=2, max=7)
        ])

        nc = NeighborhoodConfig(dims, radius=8, min_initialized=3)
        n = Neighborhood(nc, home_coordinate=Coordinate([3, 6]))

        rcm0 = self._construct_rcm(throughput=1, latency=5)
        rcm1 = self._construct_rcm(throughput=3, latency=5)

        n.coordinate_data.set_measurement(Coordinate([3, 6]), rcm0)  # home coordinate
        n.coordinate_data.increment_visit_count(Coordinate([3, 6]))

        n.coordinate_data.set_measurement(Coordinate([4, 5]), rcm1)
        n.coordinate_data.increment_visit_count(Coordinate([4, 5]))

        magnitude = 8
        new_coord = n.calculate_new_coordinate(magnitude,
                                               disable_clipping=True)
        self.assertEqual(new_coord, Coordinate([7, 2]))

    def test_all_same_throughputs(self):
        """
        Test that when all the coordinates in the neighborhood has the
        same throughputs, the step vector is zero and new coordinate is
        same as the home coordinate.
        """
        dims = SearchDimensions()
        dims.add_dimensions(0, [
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("foobar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ])

        nc = NeighborhoodConfig(dims, radius=3, min_initialized=3)
        n = Neighborhood(nc, home_coordinate=Coordinate([1, 1, 1]))

        rcm0 = self._construct_rcm(throughput=10, latency=5)
        rcm1 = self._construct_rcm(throughput=10, latency=5)
        rcm2 = self._construct_rcm(throughput=10, latency=5)
        rcm3 = self._construct_rcm(throughput=10, latency=5)

        n.coordinate_data.set_measurement(Coordinate([1, 1, 1]), rcm0)  # home coordinate
        n.coordinate_data.increment_visit_count(Coordinate([1, 1, 1]))

        n.coordinate_data.set_measurement(Coordinate([1, 0, 0]), rcm1)
        n.coordinate_data.increment_visit_count(Coordinate([1, 0, 0]))

        n.coordinate_data.set_measurement(Coordinate([0, 1, 0]), rcm2)
        n.coordinate_data.increment_visit_count(Coordinate([0, 1, 0]))

        n.coordinate_data.set_measurement(Coordinate([0, 0, 1]), rcm3)
        n.coordinate_data.increment_visit_count(Coordinate([0, 0, 1]))

        step_vector = n._get_step_vector()
        self.assertEqual(step_vector, Coordinate([0, 0, 0]))

        magnitude = 5
        new_coord = n.calculate_new_coordinate(magnitude,
                                               disable_clipping=True)
        self.assertEqual(new_coord, Coordinate([1, 1, 1]))
