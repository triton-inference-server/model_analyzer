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

from experiments.experiment_evaluator import ExperimentEvaluator
from generator_experiment_factory import GeneratorExperimentFactory
from experiment_config_command_creator import ExperimentConfigCommandCreator
from experiment_data import ExperimentData
from checkpoint_experiment_data import CheckpointExperimentData


class EvaluateConfigGenerator:
    """
    Class to run and evaluate an ConfigGenerator algorithm using an 
    existing checkpoint of raw measurement data
    """

    def __init__(self, model_name, data_path):
        self._model_name = model_name
        self._config_command = ExperimentConfigCommandCreator.make_config(
            data_path, model_name)

        self._checkpoint_data = CheckpointExperimentData(self._config_command)
        self._profile_data = ExperimentData()

    def execute_generator(self, generator_name):

        generator = GeneratorExperimentFactory.create_generator(
            generator_name, self._config_command)

        self._run_generator(generator)

    def print_results(self):
        result_evaluator = ExperimentEvaluator(self._checkpoint_data,
                                               self._profile_data)
        result_evaluator.print_results()

    def _run_generator(self, cg):
        config_generator = cg.next_config()

        while not cg.is_done():
            run_config = next(config_generator)

            run_config_measurement = self._checkpoint_data.get_run_config_measurement(
                run_config)
            self._profile_data.add_run_config_measurement(
                run_config, run_config_measurement)

            cg.set_last_results([run_config_measurement])
