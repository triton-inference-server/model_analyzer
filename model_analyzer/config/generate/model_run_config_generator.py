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

from .config_generator_interface import ConfigGeneratorInterface
from .generator_factory import ConfigGeneratorFactory
from .perf_analyzer_config_generator import PerfAnalyzerConfigGenerator

from model_analyzer.config.run.model_run_config import ModelRunConfig


class ModelRunConfigGenerator(ConfigGeneratorInterface):
    """
    Given a model, generates all ModelRunConfigs (combination of 
    ModelConfig and PerfConfig)
    """

    def __init__(self, config, gpus, model, client, variant_name_manager,
                 default_only):
        """
        Parameters
        ----------
        config: ModelAnalyzerConfig
        
        gpus: List of GPUDevices
        
        model: ConfigModelProfileSpec
            The model to generate ModelRunConfigs for
            
        client: TritonClient

        variant_name_manager: ModelVariantNameManager
        
        default_only: Bool
        """
        self._config = config
        self._gpus = gpus
        self._model = model
        self._client = client
        self._variant_name_manger = variant_name_manager

        self._model_name = model.model_name()

        self._model_pa_flags = model.perf_analyzer_flags()
        self._model_parameters = model.parameters()
        self._triton_server_env = model.triton_server_environment()

        self._determine_early_exit_enables(config, model)

        # This prevents an error when is_done() is checked befored the first next_config() call
        self._pacg = PerfAnalyzerConfigGenerator(self._config, self._model_name,
                                                 self._model_pa_flags,
                                                 self._model_parameters,
                                                 self._pacg_early_exit_enable)

        self._mcg = ConfigGeneratorFactory.create_model_config_generator(
            self._config, self._gpus, model, self._client,
            self._variant_name_manger, default_only,
            self._mcg_early_exit_enable)

        self._curr_mc_measurements = []

    def is_done(self):
        return self._pacg.is_done() and self._mcg.is_done()

    def next_config(self):
        """
        Returns
        -------
        ModelRunConfig
            The next ModelRunConfig generated by this class
        """

        model_config_generator = self._mcg.next_config()

        while not self._mcg.is_done():
            model_config = next(model_config_generator)

            self._pacg = PerfAnalyzerConfigGenerator(
                self._config, model_config.get_field('name'),
                self._model_pa_flags, self._model_parameters,
                self._pacg_early_exit_enable)
            perf_analyzer_config_generator = self._pacg.next_config()
            while not self._pacg.is_done():
                perf_analyzer_config = next(perf_analyzer_config_generator)
                run_config = self._generate_model_run_config(
                    model_config, perf_analyzer_config)
                yield run_config

    def set_last_results(self, measurements):
        """ 
        Given the results from the last ModelRunConfig, make decisions 
        about future configurations to generate

        Parameters
        ----------
        measurements: List of Measurements from the last run(s)
        """
        self._pacg.set_last_results(measurements)
        self._curr_mc_measurements.extend(measurements)
        if self._pacg.is_done():
            self._mcg.set_last_results(self._curr_mc_measurements)
            self._curr_mc_measurements = []

    def _generate_model_run_config(self, model_config, perf_analyzer_config):
        run_config = ModelRunConfig(self._model_name, model_config,
                                    perf_analyzer_config)

        return run_config

    def _determine_early_exit_enables(self, config, model):
        early_exit_enable = config.early_exit_enable
        concurrency_specified = model.parameters()['concurrency']
        config_parameters_exist = model.model_config_parameters()

        self._pacg_early_exit_enable = early_exit_enable or not concurrency_specified
        self._mcg_early_exit_enable = early_exit_enable or not config_parameters_exist
