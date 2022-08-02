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

from model_analyzer.config.generate.model_variant_name_manager import ModelVariantNameManager
from .common import test_result_collector as trc
import unittest
from unittest.mock import patch

from tests.common.test_utils import default_encode
from model_analyzer.constants import DEFAULT_CONFIG_PARAMS


class TestModelVariantNameManager(trc.TestResultCollector):

    def setUp(self):
        self._mvnm = ModelVariantNameManager()
        self._non_default_param_combo = {'foo': 1}

    def tearDown(self):
        patch.stopall()

    def test_default(self):
        """
        Check that default config is returned
        """
        name = self._mvnm.get_model_variant_name("modelA", {'A': 1},
                                                 DEFAULT_CONFIG_PARAMS)

        self.assertEqual(name, (False, "modelA_config_default"))

    def test_basic(self):
        """
        If multiple unique model configs are passed in, the name will keep
        incrementing
        """
        name0 = self._mvnm.get_model_variant_name("modelA", {'A': 1},
                                                  self._non_default_param_combo)
        name1 = self._mvnm.get_model_variant_name("modelA", {'A': 2},
                                                  self._non_default_param_combo)
        name2 = self._mvnm.get_model_variant_name("modelA", {'A': 4},
                                                  self._non_default_param_combo)

        self.assertEqual(name0, (False, "modelA_config_0"))
        self.assertEqual(name1, (False, "modelA_config_1"))
        self.assertEqual(name2, (False, "modelA_config_2"))

    def test_multiple_models(self):
        """
        The two models should have no impact on each other's naming or counts
        """

        a0 = self._mvnm.get_model_variant_name("modelA", {'A': 1},
                                               self._non_default_param_combo)
        b0 = self._mvnm.get_model_variant_name("modelB", {'A': 1},
                                               self._non_default_param_combo)
        a1 = self._mvnm.get_model_variant_name("modelA", {'A': 2},
                                               self._non_default_param_combo)
        b1 = self._mvnm.get_model_variant_name("modelB", {'A': 2},
                                               self._non_default_param_combo)

        self.assertEqual(a0, (False, "modelA_config_0"))
        self.assertEqual(a1, (False, "modelA_config_1"))
        self.assertEqual(b0, (False, "modelB_config_0"))
        self.assertEqual(b1, (False, "modelB_config_1"))

    def test_repeat(self):
        """
        Calling with the same param_combo multiple times should result
        in the same name being returned
        """

        a0 = self._mvnm.get_model_variant_name("modelA", {'A': 1},
                                               self._non_default_param_combo)
        a1 = self._mvnm.get_model_variant_name("modelA", {'A': 1},
                                               self._non_default_param_combo)

        self.assertEqual(a0, (False, "modelA_config_0"))
        self.assertEqual(a1, (True, "modelA_config_0"))

    def test_from_dict(self):
        """
        Restoring from a dict should see existing configs
        """
        _ = self._mvnm.get_model_variant_name("modelA", {'A': 1},
                                              self._non_default_param_combo)
        _ = self._mvnm.get_model_variant_name("modelB", {'A': 1},
                                              self._non_default_param_combo)
        _ = self._mvnm.get_model_variant_name("modelA", {'A': 2},
                                              self._non_default_param_combo)

        mvnm_dict = default_encode(self._mvnm)

        mvnm = ModelVariantNameManager._from_dict(mvnm_dict)

        self.assertEqual(mvnm._model_config_dicts,
                         self._mvnm._model_config_dicts)
        self.assertEqual(mvnm._model_name_index, self._mvnm._model_name_index)

        a0 = mvnm.get_model_variant_name("modelA", {'A': 1},
                                         self._non_default_param_combo)
        b0 = mvnm.get_model_variant_name("modelB", {'A': 1},
                                         self._non_default_param_combo)
        a1 = mvnm.get_model_variant_name("modelA", {'A': 2},
                                         self._non_default_param_combo)
        b1 = mvnm.get_model_variant_name("modelB", {'A': 2},
                                         self._non_default_param_combo)

        self.assertEqual(a0, (True, "modelA_config_0"))
        self.assertEqual(a1, (True, "modelA_config_1"))
        self.assertEqual(b0, (True, "modelB_config_0"))
        self.assertEqual(b1, (False, "modelB_config_1"))


if __name__ == '__main__':
    unittest.main()
