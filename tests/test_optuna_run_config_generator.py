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
from model_analyzer.config.input.config_defaults import DEFAULT_BATCH_SIZES
from model_analyzer.config.input.objects.config_model_profile_spec import (
    ConfigModelProfileSpec,
)
from model_analyzer.device.gpu_device import GPUDevice
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

        config = self._create_config()
        self._rcg = OptunaRunConfigGenerator(
            config=config,
            gpus=[GPUDevice("TEST_DEVICE_NAME", 0, "TEST_BUS_ID0", "TEST_UUID0")],
            models=self._mock_models,
            model_variant_name_manager=ModelVariantNameManager(),
            search_parameters=MagicMock(),
            seed=100,
        )

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
        self._rcg._create_trial_objectives(trial)
        run_config = self._rcg._create_objective_based_run_config()

        model_config = run_config.model_run_configs()[0].model_config()
        perf_config = run_config.model_run_configs()[0].perf_config()

        self.assertEqual(model_config.to_dict()["name"], self._test_config_dict["name"])

        # These values are the result of using a fixed seed of 100
        self.assertEqual(model_config.to_dict()["maxBatchSize"], 8)
        self.assertEqual(model_config.to_dict()["instanceGroup"][0]["count"], 5)
        self.assertEqual(perf_config["batch-size"], DEFAULT_BATCH_SIZES)
        self.assertEqual(perf_config["concurrency-range"], 80)

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
                - my-model
            """)
        # yapf: enable

        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def tearDown(self):
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
