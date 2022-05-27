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

import unittest
from unittest.mock import MagicMock, patch

from .common import test_result_collector as trc
from model_analyzer.config.generate.coordinate import Coordinate
from model_analyzer.config.generate.search_config import SearchConfig
from model_analyzer.config.generate.search_dimension import SearchDimension
from model_analyzer.config.generate.undirected_run_config_generator import UndirectedRunConfigGenerator
from model_analyzer.config.input.objects.config_model_profile_spec import ConfigModelProfileSpec


class TestUndirectedRunConfigGenerator(trc.TestResultCollector):

    def setUp(self):
        mock_models = [ConfigModelProfileSpec(model_name="fake_model_name")]
        sc = SearchConfig(dimensions=[
            SearchDimension("max_batch_size",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL),
            SearchDimension("instance_count",
                            SearchDimension.DIMENSION_TYPE_LINEAR),
            SearchDimension("concurrency",
                            SearchDimension.DIMENSION_TYPE_EXPONENTIAL)
        ],
                          neighborhood_radius=5,
                          step_magnitude=7,
                          min_initialized=2)
        self._urcg = UndirectedRunConfigGenerator(sc, MagicMock(), mock_models)

    def test_get_starting_coordinate(self):
        """ Test that get_starting_coordinate() works for non-zero values """
        #yapf: disable
        sc = SearchConfig(dimensions=[
                SearchDimension("x", SearchDimension.DIMENSION_TYPE_EXPONENTIAL, min=2),
                SearchDimension("y", SearchDimension.DIMENSION_TYPE_LINEAR, min=1),
                SearchDimension("z", SearchDimension.DIMENSION_TYPE_EXPONENTIAL, min=3)
             ])
        #yapf: enable
        urcg = UndirectedRunConfigGenerator(sc, MagicMock(), MagicMock())
        self.assertEqual(urcg._get_starting_coordinate(), Coordinate([2, 1, 3]))

    def test_get_next_run_config(self):
        """ 
        Test that get_next_run_config() creates a proper RunConfig 
        
        Sets up a case where the coordinate is [5,7,4], which cooresponds to
          - max_batch_size = 32
          - instance_count = 8
          - concurrency = 16
        Also, dynamic batching should be on, and existing values from the base model
        config should persist if they aren't overwritten
        """
        urcg = self._urcg
        urcg._coordinate_to_measure = Coordinate([5, 7, 4])

        fake_base_config = {
            "name": "fake_model_name",
            "input": [{
                "name": "INPUT__0",
                "dataType": "TYPE_FP32",
                "dims": [16]
            }],
            "max_batch_size": 4
        }

        expected_model_config = {
            'cpu_only':
                False,
            'dynamicBatching': {},
            'instanceGroup': [{
                'count': 8,
                'kind': 'KIND_GPU'
            }],
            'maxBatchSize':
                32,
            'name':
                'fake_model_name_config_0',
            'input': [{
                "name": "INPUT__0",
                "dataType": "TYPE_FP32",
                "dims": ['16']
            }]
        }

        with patch(
                "model_analyzer.config.generate.base_model_config_generator.BaseModelConfigGenerator.get_base_model_config_dict",
                return_value=fake_base_config):
            rc = urcg._get_next_run_config()

        self.assertEqual(len(rc.model_run_configs()), 1)
        model_config = rc.model_run_configs()[0].model_config()
        perf_config = rc.model_run_configs()[0].perf_config()

        self.assertEqual(model_config.to_dict(), expected_model_config)
        self.assertEqual(perf_config['concurrency-range'], 16)
        self.assertEqual(perf_config['batch-size'], 1)

    def test_radius_magnitude(self):
        """ 
        Test that get_radius and get_magnitude work correctly 

        Expected radius is 6 (search config has 5, and offset is 1)
        Expected magnitude is 8 (search config has 7, and offset is 1)
        """

        urcg = self._urcg
        urcg._radius_offset = 1
        urcg._magnitude_offset = 1

        self.assertEqual(urcg._get_radius(), 6)
        self.assertEqual(urcg._get_magnitude(), 8)

    def tearDown(self):
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
