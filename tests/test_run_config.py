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

import unittest
from unittest.mock import MagicMock

from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.triton.model.model_config import ModelConfig

from .common import test_result_collector as trc

from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.config.run.run_config import RunConfig


class TestRunConfig(trc.TestResultCollector):

    def test_triton_env(self):
        """ 
        Test triton env initialized correctly and returns correctly from function
        """
        fake_env = {'a': 5, 'b': {'c': 7}}
        rc = RunConfig(fake_env)
        self.assertEqual(rc.triton_environment(), fake_env)

    def test_model_run_configs(self):
        """
        Test adding and getting ModelRunConfigs
        """
        mrc1 = ModelRunConfig("model1", MagicMock(), MagicMock())
        mrc2 = ModelRunConfig("model2", MagicMock(), MagicMock())
        rc = RunConfig({})
        rc.add_model_run_config(mrc1)
        rc.add_model_run_config(mrc2)

        mrc_out = rc.model_run_configs()

        self.assertEqual(mrc_out[0], mrc1)
        self.assertEqual(mrc_out[1], mrc2)

    def test_representation(self):
        """
        Test that representation() is just a string join of member ModelRunConfig's representations
        """
        pc1 = PerfAnalyzerConfig()
        pc1.update_config({'model-name': "TestModel1"})
        pc2 = PerfAnalyzerConfig()
        pc2.update_config({'model-name': "TestModel2"})
        mrc1 = ModelRunConfig("model1", MagicMock(), pc1)
        mrc2 = ModelRunConfig("model2", MagicMock(), pc2)
        rc = RunConfig({})
        rc.add_model_run_config(mrc1)
        rc.add_model_run_config(mrc2)

        expected_representation = pc1.representation() + pc2.representation()
        self.assertEqual(rc.representation(), expected_representation)

    def test_cpu_only(self):
        """
        Test that cpu_only() is only true if all ModelConfigs are cpu_only() 
        """
        cpu_only_true_mc = ModelConfig({})
        cpu_only_true_mc.set_cpu_only(True)
        cpu_only_false_mc = ModelConfig({})
        cpu_only_false_mc.set_cpu_only(False)

        mrc1 = ModelRunConfig("model1", cpu_only_true_mc, MagicMock())
        mrc2 = ModelRunConfig("model2", cpu_only_false_mc, MagicMock())

        rc = RunConfig({})
        rc.add_model_run_config(mrc1)
        self.assertTrue(rc.cpu_only())

        rc.add_model_run_config(mrc2)
        self.assertFalse(rc.cpu_only())


if __name__ == '__main__':
    unittest.main()
