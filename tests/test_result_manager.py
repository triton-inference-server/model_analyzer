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

from .common import test_result_collector as trc

from unittest.mock import patch, MagicMock
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager


class TestResultManager(trc.TestResultCollector):

    def test_server_data(self):
        """
        Test that add_server_data() and get_server_only_data() 
        are effectively mirrored set/get functions
        """
        state_manager = AnalyzerStateManager(config=MagicMock(), server=None)
        result_manager = ResultManager(config=MagicMock(),
                                       state_manager=state_manager)

        server_data = {'a': 5, 'b': 7}
        result_manager.add_server_data(server_data)

        self.assertEqual(server_data, result_manager.get_server_only_data())

    def test_measurements(self):
        """
        Test add_run_config_measurement and get_model_configs_run_config_measurements
        
        Confirm that run_config_measurements are stored per-model, and then per-variant, and
        then in a list per-representation
        Confirm that the measurements can be read out via get_model_configs_run_config_measurements()
        """
        state_manager = AnalyzerStateManager(config=MagicMock(), server=None)
        result_manager = ResultManager(config=MagicMock(),
                                       state_manager=state_manager)

        fake_run_config1a = MagicMock()
        fake_run_config1a.models_name.return_value = "Model1"
        fake_run_config1a.model_variants_name.return_value = "Model1_config_default"
        fake_run_config1a.representation.return_value = "1a"

        fake_run_config1b = MagicMock()
        fake_run_config1b.models_name.return_value = "Model1"
        fake_run_config1b.model_variants_name.return_value = "Model1_config_default"
        fake_run_config1b.representation.return_value = "1b"

        fake_run_config1c = MagicMock()
        fake_run_config1c.models_name.return_value = "Model1"
        fake_run_config1c.model_variants_name.return_value = "Model1_config_0"
        fake_run_config1c.representation.return_value = "1c"

        fake_run_config2 = MagicMock()
        fake_run_config2.models_name.return_value = "Model2"
        fake_run_config2.model_variants_name.return_value = "Model2_config_default"
        fake_run_config2.representation.return_value = "2"

        rcm1a = MagicMock()
        rcm1b = MagicMock()
        rcm1c = MagicMock()
        rcm2 = MagicMock()

        result_manager.add_run_config_measurement(run_config=fake_run_config1a,
                                                  run_config_measurement=rcm1a)
        result_manager.add_run_config_measurement(run_config=fake_run_config1b,
                                                  run_config_measurement=rcm1b)
        result_manager.add_run_config_measurement(run_config=fake_run_config1c,
                                                  run_config_measurement=rcm1c)
        result_manager.add_run_config_measurement(run_config=fake_run_config2,
                                                  run_config_measurement=rcm2)

        config, measurements = result_manager.get_model_configs_run_config_measurements(
            "Model1_config_default")
        self.assertEqual(config, fake_run_config1a)
        self.assertEqual(2, len(measurements))

        config, measurements = result_manager.get_model_configs_run_config_measurements(
            "Model1_config_0")
        self.assertEqual(config, fake_run_config1c)
        self.assertEqual(1, len(measurements))

        config, measurements = result_manager.get_model_configs_run_config_measurements(
            "Model2_config_default")
        self.assertEqual(config, fake_run_config2)
        self.assertEqual(1, len(measurements))

    def tearDown(self):
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
