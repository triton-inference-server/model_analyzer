# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# from tests.common.test_utils import convert_non_gpu_metrics_to_data, \
#     convert_gpu_metrics_to_data, convert_avg_gpu_metrics_to_data, \
#     construct_perf_analyzer_config, construct_run_config_measurement, default_encode

# from model_analyzer.record.metrics_manager import MetricsManager
# from model_analyzer.result.model_config_measurement import ModelConfigMeasurement
# from model_analyzer.result.run_config_measurement import RunConfigMeasurement

# from statistics import mean

import unittest
from unittest.mock import MagicMock, patch
from .common import test_result_collector as trc

from model_analyzer.result.run_config_result import RunConfigResult


class TestRunConfigResult(trc.TestResultCollector):

    def setUp(self):
        self._construct_empty_rcr()

    def tearDown(self):
        NotImplemented

    def test_model_name(self):
        """
        Test that model_name is correctly returned
        """
        self.assertEqual(self.rcr_empty.model_name(),
                         self.rcr_empty._model_name)

    def test_configs(self):
        """
        Test that model_configs is correctly returned
        """
        self.assertEqual(self.rcr_empty.model_configs(),
                         self.rcr_empty._model_configs)

    def test_failing_measurements_empty(self):
        """
        Test that failing returns true if no measurements are added
        """
        self.assertTrue(self.rcr_empty.failing())

    def _construct_empty_rcr(self):
        self.rcr_empty = RunConfigResult(model_name=MagicMock(),
                                         model_configs=MagicMock(),
                                         comparator=MagicMock(),
                                         constraints=MagicMock())


if __name__ == '__main__':
    unittest.main()
