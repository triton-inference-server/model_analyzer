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


class TestModelVariantNameManager(trc.TestResultCollector):

    def setUp(self):
        NotImplemented

    def tearDown(self):
        patch.stopall()

    def test_default(self):
        """
        If no param is passed in, it is considered the default config
        """
        mvnm = ModelVariantNameManager()
        name = mvnm.get_model_variant_name("modelA", {})
        self.assertEqual(name, "modelA_config_default")

    def test_basic(self):
        """
        If multiple unique param combos are passed in, the name will keep
        incrementing
        """
        mvnm = ModelVariantNameManager()
        name0 = mvnm.get_model_variant_name("modelA", {'A': 1})
        name1 = mvnm.get_model_variant_name("modelA", {'A': 2})
        name2 = mvnm.get_model_variant_name("modelA", {'A': 4})
        self.assertEqual(name0, "modelA_config_0")
        self.assertEqual(name1, "modelA_config_1")
        self.assertEqual(name2, "modelA_config_2")

    def test_multiple_models(self):
        """
        The two models should have no impact on each other's naming or counts
        """
        mvnm = ModelVariantNameManager()
        a0 = mvnm.get_model_variant_name("modelA", {'A': 1})
        b0 = mvnm.get_model_variant_name("modelB", {'A': 1})
        a1 = mvnm.get_model_variant_name("modelA", {'A': 2})
        b1 = mvnm.get_model_variant_name("modelB", {'A': 2})
        self.assertEqual(a0, "modelA_config_0")
        self.assertEqual(a1, "modelA_config_1")
        self.assertEqual(b0, "modelB_config_0")
        self.assertEqual(b1, "modelB_config_1")

    def test_combos(self):
        """
        Unique combos that share some of the same data should still be considered
        unique and return new names
        """
        mvnm = ModelVariantNameManager()
        a0 = mvnm.get_model_variant_name("modelA", {'A': 1})
        a1 = mvnm.get_model_variant_name("modelA", {'B': 1})
        a2 = mvnm.get_model_variant_name("modelA", {'A': 1, 'B': 1})
        self.assertEqual(a0, "modelA_config_0")
        self.assertEqual(a1, "modelA_config_1")
        self.assertEqual(a2, "modelA_config_2")

    def test_repeat(self):
        """
        Calling with the same param_combo multiple times should result
        in the same name being returned
        """

        mvnm = ModelVariantNameManager()
        a0 = mvnm.get_model_variant_name("modelA", {'A': 1})
        a1 = mvnm.get_model_variant_name("modelA", {'A': 1})
        self.assertEqual(a0, "modelA_config_0")
        self.assertEqual(a1, "modelA_config_0")

    def test_dict_order(self):
        """
        The order of the dict doesn't matter. If the contents are the same
        then the name should be the same
        """
        mvnm = ModelVariantNameManager()
        a0 = mvnm.get_model_variant_name("modelA", {'A': 1, 'B': 1})
        a1 = mvnm.get_model_variant_name("modelA", {'B': 1, 'A': 1})
        self.assertEqual(a0, "modelA_config_0")
        self.assertEqual(a1, "modelA_config_0")

    def test_list_order(self):
        """
        The order of a list DOES matter. If the contents are the same but
        in a different order, the name should be different
        """
        mvnm = ModelVariantNameManager()
        a0 = mvnm.get_model_variant_name("modelA", {'A': 1, 'B': [1, 2, 3]})
        a1 = mvnm.get_model_variant_name("modelA", {'A': 1, 'B': [3, 2, 1]})
        self.assertEqual(a0, "modelA_config_0")
        self.assertEqual(a1, "modelA_config_1")

    def test_nested_combos(self):
        """
        Make sure that having more complicated param_combos works as expected
        """
        mvnm = ModelVariantNameManager()
        a0 = mvnm.get_model_variant_name("modelA", {
            'A': {
                'C': 5,
                'D': 6
            },
            'B': [3, 2, 1]
        })

        # Same dict for A, but different order. Should return same as A0
        a1 = mvnm.get_model_variant_name("modelA", {
            'A': {
                'D': 6,
                'C': 5
            },
            'B': [3, 2, 1]
        })

        # Different dict for A. Should return new name
        a2 = mvnm.get_model_variant_name("modelA", {
            'A': {
                'C': 5,
                'D': 7
            },
            'B': [3, 2, 1]
        })

        # Different list for B. Should return new name
        a3 = mvnm.get_model_variant_name("modelA", {
            'A': {
                'C': 5,
                'D': 6
            },
            'B': [3, 2, 0]
        })

        # Same as A. Should return same as A0
        a4 = mvnm.get_model_variant_name("modelA", {
            'A': {
                'C': 5,
                'D': 6
            },
            'B': [3, 2, 1]
        })

        self.assertEqual(a0, "modelA_config_0")
        self.assertEqual(a1, "modelA_config_0")
        self.assertEqual(a2, "modelA_config_1")
        self.assertEqual(a3, "modelA_config_2")
        self.assertEqual(a4, "modelA_config_0")

        # Complicated case with all combinations of dict/list inside of dict/list
        b0 = mvnm.get_model_variant_name(
            "modelB", {
                'A': {
                    'C': {
                        'F': 4,
                        'G': 5
                    },
                    'D': [1, 2, 3],
                    'E': "abc"
                },
                'B': [{
                    'H': 4,
                    'I': 5
                }, [4, 5, 6], "J", 7]
            })

        # Same as B0 (with some dict ordering differences). Should return same name
        b1 = mvnm.get_model_variant_name(
            "modelB", {
                'A': {
                    'E': "abc",
                    'C': {
                        'G': 5,
                        'F': 4
                    },
                    'D': [1, 2, 3]
                },
                'B': [{
                    'I': 5,
                    'H': 4
                }, [4, 5, 6], "J", 7]
            })

        self.assertEqual(b0, "modelB_config_0")
        self.assertEqual(b1, "modelB_config_0")


if __name__ == '__main__':
    unittest.main()
