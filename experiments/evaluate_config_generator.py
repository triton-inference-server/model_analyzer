#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import MagicMock, patch

from checkpoint_experiment_data import CheckpointExperimentData
from experiment_config_command_creator import ExperimentConfigCommandCreator
from experiment_data import ExperimentData
from experiment_evaluator import ExperimentEvaluator
from experiment_file_writer import ExperimentFileWriter
from generator_experiment_factory import GeneratorExperimentFactory

from model_analyzer.config.generate.model_variant_name_manager import (
    ModelVariantNameManager,
)
from model_analyzer.state.analyzer_state import AnalyzerState


class EvaluateConfigGenerator:
    """
    Class to run and evaluate an ConfigGenerator algorithm using an
    existing checkpoint of raw measurement data
    """

    def __init__(self, model_name, data_path, output_path, other_args):
        self._patch_checkpoint_load()

        self._output_path = output_path
        self._model_name = model_name
        self._config_command = ExperimentConfigCommandCreator.make_config(
            data_path, model_name, other_args
        )

        self._checkpoint_data = CheckpointExperimentData(self._config_command)
        self._profile_data = ExperimentData()

        self._default_config_dict = self._checkpoint_data.get_default_config_dict()
        p = patch(
            "model_analyzer.config.generate.base_model_config_generator.BaseModelConfigGenerator.get_base_model_config_dict",
            MagicMock(return_value=self._default_config_dict),
        )
        p.start()

    def execute_generator(self):
        generator = GeneratorExperimentFactory.create_generator(self._config_command)

        self._run_generator(generator)

    def print_results(self):
        result_evaluator = ExperimentEvaluator(
            self._checkpoint_data, self._profile_data, self._config_command
        )
        result_evaluator.print_results()

    def store_results(self):
        configs = self._config_command.get_all_config()
        file_writer = ExperimentFileWriter(
            self._output_path, file_name=f"output_{self._model_name}.csv"
        )
        file_writer.write(
            self._checkpoint_data,
            self._profile_data,
            configs["radius"],
            configs["min_initialized"],
        )

    def _run_generator(self, cg):
        for run_config in cg.get_configs():
            run_config_measurement = self._checkpoint_data.get_run_config_measurement(
                run_config
            )

            if run_config_measurement:
                run_config_measurement.set_metric_weightings(
                    metric_objectives=[self._config_command.objectives]
                )
                run_config_measurement.set_model_config_constraints(
                    model_config_constraints=[self._config_command.constraints]
                )

            self._profile_data.add_run_config_measurement(
                run_config, run_config_measurement
            )

            cg.set_last_results([run_config_measurement])

    def _patch_checkpoint_load(self):
        old_fn = AnalyzerState.from_dict

        def patched_analyzer_state_from_dict(state_dict):
            if "ModelManager.model_variant_name_manager" not in state_dict:
                state_dict[
                    "ModelManager.model_variant_name_manager"
                ] = ModelVariantNameManager()
            return old_fn(state_dict)

        p = patch(
            "model_analyzer.state.analyzer_state.AnalyzerState.from_dict",
            patched_analyzer_state_from_dict,
        )
        p.start()
