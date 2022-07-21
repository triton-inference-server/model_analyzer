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

from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from model_analyzer.config.generate.coordinate_data import CoordinateData, NeighborhoodData
from model_analyzer.config.generate.coordinate import Coordinate

from .common.test_utils import construct_run_config_measurement
from .common import test_result_collector as trc


class TestCoordinateData(trc.TestResultCollector):

    def _construct_rcm(self,
                       throughput: float,
                       latency: float,
                       config_name: str = "modelA_config_0"):
        model_name = "modelA"
        model_config_name = [config_name]

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

    def test_basic(self):
        result_data = CoordinateData()

        coordinate = Coordinate([0, 0, 0])
        self.assertEqual(result_data.get_measurement(coordinate), None)
        self.assertEqual(result_data.get_visit_count(coordinate), 0)

        neighborhood_data = NeighborhoodData()
        self.assertEqual(neighborhood_data.get_measurement(coordinate), None)

    def test_visit_count(self):
        result_data = CoordinateData()

        coordinate1 = Coordinate([0, 0, 0])
        coordinate2 = Coordinate([0, 4, 1])

        result_data.increment_visit_count(coordinate1)
        self.assertEqual(1, result_data.get_visit_count(coordinate1))

        result_data.increment_visit_count(coordinate2)
        self.assertEqual(1, result_data.get_visit_count(coordinate2))

        result_data.increment_visit_count(coordinate1)
        result_data.increment_visit_count(coordinate1)
        self.assertEqual(3, result_data.get_visit_count(coordinate1))
        self.assertEqual(1, result_data.get_visit_count(coordinate2))

    def test_neighborhood_measurement(self):
        """
        Test if NeighborhoodData can properly set and get and reset
        the measurements correctly.
        """
        neighborhood_data = NeighborhoodData()

        coordinate1 = Coordinate([0, 0, 0])
        coordinate2 = Coordinate([0, 4, 1])

        rcm0 = self._construct_rcm(10, 5, config_name="modelA_config_0")
        rcm1 = self._construct_rcm(10, 5, config_name="modelB_config_0")

        neighborhood_data.set_measurement(coordinate1, rcm0)
        neighborhood_rcm0 = neighborhood_data.get_measurement(coordinate1)
        self.assertEqual(rcm0.model_variants_name(),
                         neighborhood_rcm0.model_variants_name())

        neighborhood_data.set_measurement(coordinate2, rcm1)
        neighborhood_rcm1 = neighborhood_data.get_measurement(coordinate2)
        self.assertEqual(rcm1.model_variants_name(),
                         neighborhood_rcm1.model_variants_name())

        neighborhood_data.reset_measurements()
        self.assertEqual(None, neighborhood_data.get_measurement(coordinate1))
        self.assertEqual(None, neighborhood_data.get_measurement(coordinate2))
