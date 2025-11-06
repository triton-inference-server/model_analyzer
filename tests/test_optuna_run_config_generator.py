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
from model_analyzer.config.input.config_defaults import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_RUN_CONFIG_MIN_REQUEST_RATE,
)
from model_analyzer.config.input.objects.config_model_profile_spec import (
    ConfigModelProfileSpec,
)
from tests.common.test_utils import evaluate_mock_config
from tests.test_config import TestConfig

from .common import test_result_collector as trc
from .mocks.mock_model_config import MockModelConfig


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
        mock_model_config = MockModelConfig("max_batch_size: 8")
        mock_model_config.start()
        model = ModelProfileSpec(
            config.profile_models[0], config, MagicMock(), MagicMock()
        )
        search_parameters = SearchParameters(model=model, config=config)
        mock_model_config.stop()

        self._rcg = OptunaRunConfigGenerator(
            config=config,
            state_manager=MagicMock(),
            gpu_count=1,
            models=self._mock_models,
            composing_models=[],
            model_variant_name_manager=ModelVariantNameManager(),
            search_parameters={"add_sub": search_parameters},
            composing_search_parameters={},
            user_seed=100,
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
        self.assertEqual(max_configs_to_search, 12)

    def test_max_number_of_configs_to_search_count(self):
        """
        Test count based max num of configs to search
        """
        config = self._create_config(additional_args=["--optuna-max-trials", "6"])

        self._rcg._config = config

        max_configs_to_search = (
            self._rcg._determine_maximum_number_of_configs_to_search()
        )

        self.assertEqual(max_configs_to_search, 6)

    def test_max_number_of_configs_to_search_both(self):
        """
        Test max count based on specify both a count and percentage
        """
        config = self._create_config(
            additional_args=[
                "--optuna-max-trials",
                "6",
                "--max-percentage-of-search-space",
                "3",
            ]
        )

        self._rcg._config = config

        max_configs_to_search = (
            self._rcg._determine_maximum_number_of_configs_to_search()
        )

        # Since both are specified we will use the smaller of the two (3% of 120 = 3)
        self.assertEqual(max_configs_to_search, 3)

    def test_min_number_of_configs_to_search_percentage(self):
        """
        Test percentage based min num of configs to search
        """
        min_configs_to_search = (
            self._rcg._determine_minimum_number_of_configs_to_search()
        )

        # Batch sizes (8) * Instance groups (5) * queue delays (3) = 120
        # 5% of search space (120) = 6
        self.assertEqual(min_configs_to_search, 6)

    def test_min_number_of_configs_to_search_count(self):
        """
        Test count based min num of configs to search
        """
        config = self._create_config(additional_args=["--optuna-min-trials", "12"])

        self._rcg._config = config

        min_configs_to_search = (
            self._rcg._determine_minimum_number_of_configs_to_search()
        )

        self.assertEqual(min_configs_to_search, 12)

    def test_min_number_of_configs_to_search_both(self):
        """
        Test min count based on specify both a count and percentage
        """
        config = self._create_config(
            additional_args=[
                "--optuna-min-trials",
                "6",
                "--min-percentage-of-search-space",
                "3",
            ]
        )

        self._rcg._config = config

        min_configs_to_search = (
            self._rcg._determine_minimum_number_of_configs_to_search()
        )

        # Since both are specified we will use the larger of the two (trials=6)
        self.assertEqual(min_configs_to_search, 6)

    def test_create_default_run_config_with_concurrency(self):
        """
        Test that a default run config with concurrency is properly created
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

    def test_create_default_run_config_with_request_rate(self):
        """
        Test that a default run config with request rate is properly created
        """
        config = self._create_config(["--request-rate-search-enable"])
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        model = ModelProfileSpec(
            config.profile_models[0], config, MagicMock(), MagicMock()
        )
        mock_model_config.stop()
        search_parameters = SearchParameters(
            model=model,
            config=config,
        )

        rcg = OptunaRunConfigGenerator(
            config=config,
            state_manager=MagicMock(),
            gpu_count=1,
            models=self._mock_models,
            composing_models=[],
            model_variant_name_manager=ModelVariantNameManager(),
            search_parameters={"add_sub": search_parameters},
            composing_search_parameters={},
            user_seed=100,
        )

        default_run_config = rcg._create_default_run_config()
        self.assertEqual(len(default_run_config.model_run_configs()), 1)

        model_config = default_run_config.model_run_configs()[0].model_config()
        perf_config = default_run_config.model_run_configs()[0].perf_config()

        self.assertEqual(model_config.to_dict()["name"], self._test_config_dict["name"])
        self.assertEqual(perf_config["batch-size"], DEFAULT_BATCH_SIZES)
        self.assertEqual(
            perf_config["request-rate-range"], DEFAULT_RUN_CONFIG_MIN_REQUEST_RATE
        )
        self.assertEqual(perf_config["concurrency-range"], None)

    def test_create_objective_based_run_config_with_concurrency(self):
        """
        Test that an objective based run config with concurrency is properly created
        """
        trial = self._rcg._study.ask()
        trial_objectives = self._rcg._create_trial_objectives(trial)
        run_config = self._rcg._create_objective_based_run_config(
            trial_objectives, None
        )

        model_config = run_config.model_run_configs()[0].model_config()
        perf_config = run_config.model_run_configs()[0].perf_config()

        self.assertEqual(model_config.to_dict()["name"], self._test_config_dict["name"])

        # These values are the result of using a fixed user_seed of 100
        self.assertEqual(model_config.to_dict()["maxBatchSize"], 16)
        self.assertEqual(model_config.to_dict()["instanceGroup"][0]["count"], 2)
        self.assertEqual(
            model_config.to_dict()["dynamicBatching"]["maxQueueDelayMicroseconds"],
            "200",
        )
        self.assertEqual(perf_config["batch-size"], DEFAULT_BATCH_SIZES)
        self.assertEqual(perf_config["concurrency-range"], 64)

    def test_create_objective_based_run_config_with_request_rate(self):
        """
        Test that an objective based run config with request rate is properly created
        """
        config = self._create_config(["--request-rate", "1024,2048"])
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        model = ModelProfileSpec(
            config.profile_models[0], config, MagicMock(), MagicMock()
        )
        mock_model_config.stop()
        search_parameters = SearchParameters(
            model=model,
            config=config,
        )

        rcg = OptunaRunConfigGenerator(
            config=config,
            state_manager=MagicMock(),
            gpu_count=1,
            models=self._mock_models,
            composing_models=[],
            model_variant_name_manager=ModelVariantNameManager(),
            search_parameters={"add_sub": search_parameters},
            composing_search_parameters={},
            user_seed=100,
        )

        trial = rcg._study.ask()
        trial_objectives = rcg._create_trial_objectives(trial)
        run_config = rcg._create_objective_based_run_config(trial_objectives, None)

        model_config = run_config.model_run_configs()[0].model_config()
        perf_config = run_config.model_run_configs()[0].perf_config()

        # These values are the result of using a fixed seed of 100
        self.assertEqual(model_config.to_dict()["name"], self._test_config_dict["name"])
        self.assertEqual(perf_config["batch-size"], DEFAULT_BATCH_SIZES)
        self.assertEqual(perf_config["request-rate-range"], 2048)
        self.assertEqual(perf_config["concurrency-range"], None)

    def test_create_run_config_with_concurrency_formula(self):
        """
        Tests that the concurrency formula option is used correctly
        """
        config = self._create_config(["--use-concurrency-formula"])
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        model = ModelProfileSpec(
            config.profile_models[0], config, MagicMock(), MagicMock()
        )
        mock_model_config.stop()
        search_parameters = SearchParameters(
            model=model,
            config=config,
        )

        rcg = OptunaRunConfigGenerator(
            config=config,
            state_manager=MagicMock(),
            gpu_count=1,
            models=self._mock_models,
            composing_models=[],
            model_variant_name_manager=ModelVariantNameManager(),
            search_parameters={"add_sub": search_parameters},
            composing_search_parameters={},
            user_seed=100,
        )

        trial = rcg._study.ask()
        trial_objectives = rcg._create_trial_objectives(trial)
        run_config = rcg._create_objective_based_run_config(trial_objectives, None)

        model_config = run_config.model_run_configs()[0].model_config()
        perf_config = run_config.model_run_configs()[0].perf_config()

        self.assertEqual(model_config.to_dict()["name"], self._test_config_dict["name"])

        # These values are the result of using a fixed user_seed of 100
        self.assertEqual(model_config.to_dict()["maxBatchSize"], 16)
        self.assertEqual(model_config.to_dict()["instanceGroup"][0]["count"], 2)
        self.assertEqual(
            model_config.to_dict()["dynamicBatching"]["maxQueueDelayMicroseconds"],
            "200",
        )
        self.assertEqual(perf_config["batch-size"], DEFAULT_BATCH_SIZES)
        self.assertEqual(perf_config["concurrency-range"], 64)

    def test_create_run_bls_config(self):
        """
        Tests that a BLS run config is created correctly
        """
        config = self._create_bls_config()
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        bls_model = ModelProfileSpec(
            config.profile_models[0], config, MagicMock(), MagicMock()
        )
        add_model = ModelProfileSpec(
            config.bls_composing_models[0], config, MagicMock(), MagicMock()
        )
        sub_model = ModelProfileSpec(
            config.bls_composing_models[1], config, MagicMock(), MagicMock()
        )
        mock_model_config.stop()
        bls_search_parameters = SearchParameters(
            model=bls_model,
            config=config,
        )
        add_search_parameters = SearchParameters(
            model=add_model, config=config, is_composing_model=True
        )
        sub_search_parameters = SearchParameters(
            model=sub_model, config=config, is_composing_model=True
        )
        rcg = OptunaRunConfigGenerator(
            config=config,
            state_manager=MagicMock(),
            gpu_count=1,
            models=[bls_model],
            composing_models=[add_model, sub_model],
            model_variant_name_manager=ModelVariantNameManager(),
            search_parameters={"bls": bls_search_parameters},
            composing_search_parameters={
                "add": add_search_parameters,
                "sub": sub_search_parameters,
            },
            user_seed=100,
        )

        trial = rcg._study.ask()
        trial_objectives = rcg._create_trial_objectives(trial)
        composing_trial_objectives = rcg._create_composing_trial_objectives(trial)
        run_config = rcg._create_objective_based_run_config(
            trial_objectives, composing_trial_objectives
        )

        bls_model_config = run_config.model_run_configs()[0].model_config()
        add_model_config = run_config.model_run_configs()[0].composing_configs()[0]
        sub_model_config = run_config.model_run_configs()[0].composing_configs()[1]
        perf_config = run_config.model_run_configs()[0].perf_config()

        # BLS (Top Level Model) + PA Config (user_seed=100)
        # =====================================================================
        self.assertEqual(bls_model_config.to_dict()["name"], "bls")
        self.assertEqual(bls_model_config.to_dict()["instanceGroup"][0]["count"], 3)
        self.assertEqual(perf_config["batch-size"], DEFAULT_BATCH_SIZES)
        self.assertEqual(perf_config["concurrency-range"], 8)

        # ADD (composing model)
        # =====================================================================
        self.assertEqual(add_model_config.to_dict()["name"], "add")
        self.assertEqual(add_model_config.to_dict()["instanceGroup"][0]["count"], 3)
        self.assertEqual(
            add_model_config.to_dict()["dynamicBatching"]["maxQueueDelayMicroseconds"],
            "100",
        )

        # SUB (composing model)
        # =====================================================================
        self.assertEqual(sub_model_config.to_dict()["name"], "sub")
        self.assertEqual(sub_model_config.to_dict()["instanceGroup"][0]["count"], 4)
        self.assertEqual(
            sub_model_config.to_dict()["dynamicBatching"]["maxQueueDelayMicroseconds"],
            "400",
        )

    def test_create_run_multi_model_config(self):
        """
        Tests that a multi-model run config is created correctly
        """
        config = self._create_multi_model_config()
        mock_model_config = MockModelConfig()
        mock_model_config.start()
        add_model = ModelProfileSpec(
            config.profile_models[0], config, MagicMock(), MagicMock()
        )
        vgg_model = ModelProfileSpec(
            config.profile_models[1], config, MagicMock(), MagicMock()
        )
        mock_model_config.stop()
        add_search_parameters = SearchParameters(
            model=add_model,
            config=config,
        )
        vgg_search_parameters = SearchParameters(
            model=vgg_model,
            config=config,
        )
        rcg = OptunaRunConfigGenerator(
            config=config,
            state_manager=MagicMock(),
            gpu_count=1,
            models=[add_model, vgg_model],
            composing_models=[],
            model_variant_name_manager=ModelVariantNameManager(),
            search_parameters={
                "add_sub": add_search_parameters,
                "vgg19_libtorch": vgg_search_parameters,
            },
            composing_search_parameters={},
            user_seed=100,
        )

        trial = rcg._study.ask()
        trial_objectives = rcg._create_trial_objectives(trial)
        composing_trial_objectives = rcg._create_composing_trial_objectives(trial)
        run_config = rcg._create_objective_based_run_config(
            trial_objectives, composing_trial_objectives
        )

        add_model_config = run_config.model_run_configs()[0].model_config()
        vgg_model_config = run_config.model_run_configs()[1].model_config()
        add_perf_config = run_config.model_run_configs()[0].perf_config()
        vgg_perf_config = run_config.model_run_configs()[0].perf_config()

        # ADD_SUB + PA Config (user_seed=100)
        # =====================================================================
        self.assertEqual(add_model_config.to_dict()["name"], "add_sub")
        self.assertEqual(add_model_config.to_dict()["maxBatchSize"], 16)
        self.assertEqual(add_model_config.to_dict()["instanceGroup"][0]["count"], 2)
        self.assertEqual(
            add_model_config.to_dict()["dynamicBatching"]["maxQueueDelayMicroseconds"],
            "100",
        )
        self.assertEqual(add_perf_config["batch-size"], DEFAULT_BATCH_SIZES)
        self.assertEqual(add_perf_config["concurrency-range"], 16)

        # VGG19_LIBTORCH + PA Config (user_seed=100)
        # =====================================================================
        self.assertEqual(vgg_model_config.to_dict()["name"], "vgg19_libtorch")
        self.assertEqual(vgg_model_config.to_dict()["instanceGroup"][0]["count"], 4)
        self.assertEqual(
            vgg_model_config.to_dict()["dynamicBatching"]["maxQueueDelayMicroseconds"],
            "600",
        )
        self.assertEqual(vgg_perf_config["batch-size"], DEFAULT_BATCH_SIZES)
        self.assertEqual(vgg_perf_config["concurrency-range"], 16)

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

        yaml_str = """
            profile_models:
                add_sub:
                    model_config_parameters:
                        dynamic_batching:
                            max_queue_delay_microseconds: [100, 200, 300]

            """

        config = TestConfig()._evaluate_config(args, yaml_str)

        return config

    def _create_bls_config(self, additional_args=[]):
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

        yaml_str = """
            profile_models: bls
            bls_composing_models:
              add:
                model_config_parameters:
                  dynamic_batching:
                    max_queue_delay_microseconds: [100, 200, 300]
              sub:
                model_config_parameters:
                  dynamic_batching:
                    max_queue_delay_microseconds: [400, 500, 600]

            """

        config = TestConfig()._evaluate_config(args, yaml_str)

        return config

    def _create_multi_model_config(self, additional_args=[]):
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

        yaml_str = """
            profile_models:
                add_sub:
                    model_config_parameters:
                        dynamic_batching:
                            max_queue_delay_microseconds: [100, 200, 300]
                vgg19_libtorch:
                    model_config_parameters:
                        dynamic_batching:
                            max_queue_delay_microseconds: [400, 500, 600]
            """

        config = TestConfig()._evaluate_config(args, yaml_str)

        return config

    def tearDown(self):
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
