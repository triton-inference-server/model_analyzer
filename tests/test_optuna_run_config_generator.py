#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.config.generate.model_profile_spec import ModelProfileSpec
from model_analyzer.config.generate.model_variant_name_manager import (
    ModelVariantNameManager,
)
from model_analyzer.config.generate.optuna_run_config_generator import (
    OptunaRunConfigGenerator,
)
from model_analyzer.config.generate.search_parameters import SearchParameters
from model_analyzer.config.input.config_defaults import DEFAULT_BATCH_SIZES
from model_analyzer.config.input.objects.config_model_profile_spec import (
    ConfigModelProfileSpec,
)
from tests.common.test_utils import evaluate_mock_config

from .common import test_result_collector as trc


class TestOptunaRunConfigGenerator(trc.TestResultCollector):
    def setUp(self):
        self._default_max_batch_size = 4
        self._test_config_dict = {
            "name": "add_sub",
            "input": [{"name": "INPUT__0", "dataType": "TYPE_FP32", "dims": [16]}],
            "max_batch_size": self._default_max_batch_size,
        }
        with patch(
            "model_analyzer.triton.model.model_config.ModelConfig.create_model_config_dict",
            return_value=self._test_config_dict,
        ):
            self._mock_models = [
                ModelProfileSpec(
                    ConfigModelProfileSpec(model_name="add_sub"),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                )
            ]

        config = self._create_config(additional_args=["--use-concurrency-formula"])
        model = config.profile_models[0]
        search_parameters = SearchParameters(
            config=config,
            parameters={},
            model_config_parameters=model.model_config_parameters(),
        )

        self._rcg = OptunaRunConfigGenerator(
            config=config,
            gpu_count=1,
            models=self._mock_models,
            model_variant_name_manager=ModelVariantNameManager(),
            search_parameters={"add_sub": search_parameters},
            seed=100,
        )

    def test_max_number_of_configs_to_search_percentage(self):
        """
        Test percentage based max num of configs to search
        """
        max_configs_to_search = (
            self._rcg._determine_maximum_number_of_configs_to_search()
        )

        # Batch sizes (8) * Instance groups (5) * queue delays (3) = 120
        # 10% of search space (120) = 12
        self.assertEquals(max_configs_to_search, 12)

    def test_max_number_of_configs_to_search_count(self):
        """
        Test count based max num of configs to search
        """
        config = self._create_config(additional_args=["--optuna_max_trials", "6"])

        self._rcg._config = config

        max_configs_to_search = (
            self._rcg._determine_maximum_number_of_configs_to_search()
        )

        self.assertEquals(max_configs_to_search, 6)

    def test_max_number_of_configs_to_search_both(self):
        """
        Test max count based on specify both a count and percentage
        """
        config = self._create_config(
            additional_args=[
                "--optuna_max_trials",
                "6",
                "--max_percentage_of_search_space",
                "3",
            ]
        )

        self._rcg._config = config

        max_configs_to_search = (
            self._rcg._determine_maximum_number_of_configs_to_search()
        )

        # Since both are specified we will use the smaller of the two (3% of 120 = 3)
        self.assertEquals(max_configs_to_search, 3)

    def test_min_number_of_configs_to_search_percentage(self):
        """
        Test percentage based min num of configs to search
        """
        min_configs_to_search = (
            self._rcg._determine_minimum_number_of_configs_to_search()
        )

        # Batch sizes (8) * Instance groups (5) * queue delays (3) = 120
        # 5% of search space (120) = 6
        self.assertEquals(min_configs_to_search, 6)

    def test_min_number_of_configs_to_search_count(self):
        """
        Test count based min num of configs to search
        """
        config = self._create_config(additional_args=["--optuna_min_trials", "12"])

        self._rcg._config = config

        min_configs_to_search = (
            self._rcg._determine_minimum_number_of_configs_to_search()
        )

        self.assertEquals(min_configs_to_search, 12)

    def test_min_number_of_configs_to_search_both(self):
        """
        Test min count based on specify both a count and percentage
        """
        config = self._create_config(
            additional_args=[
                "--optuna_min_trials",
                "6",
                "--min_percentage_of_search_space",
                "3",
            ]
        )

        self._rcg._config = config

        min_configs_to_search = (
            self._rcg._determine_minimum_number_of_configs_to_search()
        )

        # Since both are specified we will use the larger of the two (trials=6)
        self.assertEquals(min_configs_to_search, 6)

    def test_create_default_run_config(self):
        """
        Test that a default run config is properly created
        """
        default_run_config = self._rcg._create_default_run_config()

        self.assertEqual(len(default_run_config.model_run_configs()), 1)
        model_config = default_run_config.model_run_configs()[0].model_config()
        perf_config = default_run_config.model_run_configs()[0].perf_config()

        self.assertEqual(model_config.to_dict()["name"], self._test_config_dict["name"])
        self.assertEqual(perf_config["batch-size"], DEFAULT_BATCH_SIZES)
        self.assertEqual(
            perf_config["concurrency-range"], 2 * self._default_max_batch_size
        )

    def test_create_objective_based_run_config(self):
        """
        Test that an objective based run config is properly created
        """
        trial = self._rcg._study.ask()
        trial_objectives = self._rcg._create_trial_objectives(trial)
        run_config = self._rcg._create_objective_based_run_config(trial_objectives)

        model_config = run_config.model_run_configs()[0].model_config()
        perf_config = run_config.model_run_configs()[0].perf_config()

        self.assertEqual(model_config.to_dict()["name"], self._test_config_dict["name"])

        # These values are the result of using a fixed seed of 100
        self.assertEqual(model_config.to_dict()["maxBatchSize"], 16)
        self.assertEqual(model_config.to_dict()["instanceGroup"][0]["count"], 2)
        self.assertEqual(
            model_config.to_dict()["dynamicBatching"]["maxQueueDelayMicroseconds"],
            "200",
        )
        self.assertEqual(perf_config["batch-size"], DEFAULT_BATCH_SIZES)
        self.assertEqual(perf_config["concurrency-range"], 64)

    def test_create_run_config_with_concurrency_formula(self):
        config = self._create_config(["--use-concurrency-formula"])
        model = config.profile_models[0]
        search_parameters = SearchParameters(
            config=config,
            parameters={},
            model_config_parameters=model.model_config_parameters(),
        )

        rcg = OptunaRunConfigGenerator(
            config=config,
            gpu_count=1,
            models=self._mock_models,
            model_variant_name_manager=ModelVariantNameManager(),
            search_parameters={"add_sub": search_parameters},
            seed=100,
        )

        trial = rcg._study.ask()
        trial_objectives = rcg._create_trial_objectives(trial)
        run_config = rcg._create_objective_based_run_config(trial_objectives)

        model_config = run_config.model_run_configs()[0].model_config()
        perf_config = run_config.model_run_configs()[0].perf_config()

        self.assertEqual(model_config.to_dict()["name"], self._test_config_dict["name"])

        # These values are the result of using a fixed seed of 100
        self.assertEqual(model_config.to_dict()["maxBatchSize"], 16)
        self.assertEqual(model_config.to_dict()["instanceGroup"][0]["count"], 2)
        self.assertEqual(
            model_config.to_dict()["dynamicBatching"]["maxQueueDelayMicroseconds"],
            "200",
        )
        self.assertEqual(perf_config["batch-size"], DEFAULT_BATCH_SIZES)
        self.assertEqual(perf_config["concurrency-range"], 64)

    def _create_config(self, additional_args=[]):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "/tmp",
            "--config-file",
            "/tmp/my_config.yml",
        ]

        for arg in additional_args:
            args.append(arg)

        # yapf: disable
        yaml_str = ("""
            profile_models:
                add_sub:
                    model_config_parameters:
                        dynamic_batching:
                            max_queue_delay_microseconds: [100, 200, 300]

            """)
        # yapf: enable

        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def tearDown(self):
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
