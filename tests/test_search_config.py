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

from model_analyzer.config.generate.search_config import SearchConfig
from model_analyzer.config.generate.search_dimension import SearchDimension
from .common import test_result_collector as trc


class TestSearchConfig(trc.TestResultCollector):

    def test_basic(self):
        sc = SearchConfig([])
        self.assertEqual(0, sc.get_num_dimensions())

    def test_config(self):
        dimensions = []
        dimensions.append(
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR))
        dimensions.append(
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL))

        sc = SearchConfig(dimensions)

        self.assertEqual(2, sc.get_num_dimensions())

        self.assertEqual("foo", sc.get_dimension(0).get_name())
        self.assertEqual("bar", sc.get_dimension(1).get_name())

        self.assertEqual(7, sc.get_dimension(0).get_value_at_idx(6))
        self.assertEqual(64, sc.get_dimension(1).get_value_at_idx(6))

    def test_get_min_dimension(self):
        dimensions = []
        dimensions.append(
            SearchDimension("foo", SearchDimension.DIMENSION_TYPE_LINEAR, 1,
                            10))
        dimensions.append(
            SearchDimension("bar", SearchDimension.DIMENSION_TYPE_EXPONENTIAL))
        sc = SearchConfig(dimensions)

        self.assertEqual([1, 0], sc.get_min_dimension())
