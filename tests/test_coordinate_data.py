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

from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from model_analyzer.config.generate.coordinate_data import CoordinateData, NeighborhoodData
from model_analyzer.config.generate.coordinate import Coordinate

from .common.test_utils import convert_gpu_metrics_to_data
from .common import test_result_collector as trc


class TestCoordinateData(trc.TestResultCollector):

    def test_basic(self):
        result_data = CoordinateData()

        coordinate = Coordinate([0, 0, 0])
        self.assertEqual(result_data.get_throughput(coordinate), None)
        self.assertEqual(result_data.get_visit_count(coordinate), 0)

        neighborhood_data = NeighborhoodData()
        self.assertEqual(neighborhood_data.get_measurement(coordinate), None)

    def test_throughput(self):
        result_data = CoordinateData()

        coordinate1 = Coordinate([0, 0, 0])
        coordinate2 = Coordinate([0, 4, 1])

        result_data.set_throughput(coordinate1, 7)
        result_data.set_throughput(coordinate2, 9)

        self.assertEqual(7, result_data.get_throughput(coordinate1))
        self.assertEqual(9, result_data.get_throughput(coordinate2))

        # Overwrite
        result_data.set_throughput(coordinate2, 12)
        self.assertEqual(12, result_data.get_throughput(coordinate2))

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
        neighborhood_data = NeighborhoodData()

        coordinate1 = Coordinate([0, 0, 0])
        coordinate2 = Coordinate([0, 4, 1])

        gpu_data = convert_gpu_metrics_to_data({
            '0': {
                "gpu_used_memory": 10,
                "gpu_utilization": 30
            }
        })
        rcm1 = RunConfigMeasurement("model_config_0", gpu_data)
        rcm2 = RunConfigMeasurement("model_config_1", gpu_data)

        neighborhood_data.set_measurement(coordinate1, rcm1)
        neighborhood_rcm1 = neighborhood_data.get_measurement(coordinate1)
        self.assertEqual(rcm1.model_variants_name(),
                         neighborhood_rcm1.model_variants_name())

        neighborhood_data.set_measurement(coordinate2, rcm2)
        neighborhood_rcm2 = neighborhood_data.get_measurement(coordinate2)
        self.assertEqual(rcm2.model_variants_name(),
                         neighborhood_rcm2.model_variants_name())

        neighborhood_data.reset_measurements()
        self.assertEqual(None, neighborhood_data.get_measurement(coordinate1))
        self.assertEqual(None, neighborhood_data.get_measurement(coordinate2))
