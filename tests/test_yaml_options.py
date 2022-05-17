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

from .common import test_result_collector as trc
from model_analyzer.config.input.yaml_config_validator import YamlConfigValidator


class TestYamlOptions(trc.TestResultCollector):

    @classmethod
    def setUpClass(cls):
        cls.yaml_arg_validator = YamlConfigValidator()

    def test_correct_option(self):
        correct_option = "client_max_retries"
        self.assertTrue(
            TestYamlOptions.yaml_arg_validator.is_valid_option(correct_option))

    def test_misspelled_option(self):
        misspelled_option = "profile_model"  # should be "profile_models"
        self.assertFalse(
            TestYamlOptions.yaml_arg_validator.is_valid_option(
                misspelled_option))

    def test_using_hyphens_not_underscores(self):
        hyphen_option = "triton-server-flags"
        self.assertFalse(
            TestYamlOptions.yaml_arg_validator.is_valid_option(hyphen_option))

    def test_multiple_options(self):
        options = {"client_max_retries", "profile_model", "triton-server-flags"}
        count = 0
        for entry in options:
            if TestYamlOptions.yaml_arg_validator.is_valid_option(entry):
                count += 1
        self.assertEqual(count, 1,
                         "Incorrect number of yaml options returned True")
