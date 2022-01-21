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
from .model_config_generator import ModelConfigGenerator
from .perf_analyzer_config_generator import PerfAnalyzerConfigGenerator

from model_analyzer.config.run.run_config import RunConfig


class RunConfigGenerator(ConfigGeneratorInterface):
    """
    Generates all RunConfigs to execute given a list of models
    """

    def __init__(self, config, models, client):
        """
        Parameters
        ----------
        config: ModelAnalyzerConfig
        
        models: List of ConfigModelProfileSpec
            The list of models to generate RunConfigs for
            
        client: TritonClient
        """
        self._config = config
        self._models = models
        self._client = client

        self._model_names = [m.model_name() for m in models]

        # MM-PHASE 1: Assuming that all models are identical, so using first model's flag/parameters/env
        self._model_pa_flags = models[0].perf_analyzer_flags()
        self._model_parameters = models[0].parameters()
        self._triton_server_env = self._models[0].triton_server_environment()

        # This prevents an error when is_done() is checked befored the first next_config() call
        self._pacg = PerfAnalyzerConfigGenerator(self._config,
                                                 self._model_names,
                                                 self._model_pa_flags,
                                                 self._model_parameters)

        self._model_configs_are_on_final_iteration = False

    def is_done(self):
        return self._pacg.is_done(
        ) and self._model_configs_are_on_final_iteration

    def next_config(self):
        """
        Returns
        -------
        RunConfig
            The next RunConfig generated by this class
        """

        model_configs_list = self._generate_all_model_config_permuations(
            self._models)

        for model_configs in model_configs_list:
            self._pacg = PerfAnalyzerConfigGenerator(self._config,
                                                     self._model_names,
                                                     self._model_pa_flags,
                                                     self._model_parameters)

            while not self._pacg.is_done():
                perf_analyzer_config = self._pacg.next_config()
                run_config = self._create_run_config(model_configs,
                                                     perf_analyzer_config)

                self._model_configs_are_on_final_iteration = (
                    model_configs == model_configs_list[-1])

                yield run_config

    def set_last_results(self, measurements):
        """ 
        Given the results from the last RunConfig, make decisions 
        about future configurations to generate

        Parameters
        ----------
        measurements: List of Measurements from the last run(s)
        """
        self._pacg.set_last_results(measurements)

    def _create_run_config(self, model_configs, perf_analyzer_config):
        if isinstance(model_configs, list):
            run_config = self._generate_run_config(model_configs,
                                                   perf_analyzer_config)
        else:
            run_config = self._generate_run_config([model_configs],
                                                   perf_analyzer_config)
        return run_config

    def _generate_run_config(self, model_configs, perf_analyzer_config):
        run_config = RunConfig(self._model_names, model_configs,
                               perf_analyzer_config, self._triton_server_env)

        return run_config

    def _generate_all_model_config_permuations(self, models):
        """ 
        Recursively iterates through the list of models to 
        return a list of all possible model config permuations
        
        Parameters
        ----------
        models: List of ConfigModelProfileSpec
        
        Returns
        -------
        model_configs: List of ModelConfigs
        """
        child_model_configs = []
        if (len(models) > 1):
            child_model_configs.extend(
                self._generate_all_model_config_permuations(models[1:]))

        parent_model_configs = self._generate_parent_model_configs(models[0])

        return self._combine_model_config_permuations(parent_model_configs,
                                                      child_model_configs)

    def _generate_parent_model_configs(self, model):
        mcg = ModelConfigGenerator(self._config, model, self._client)

        model_configs = []
        while not mcg.is_done():
            model_configs.append(mcg.next_config())

        return model_configs

    def _combine_model_config_permuations(self, parent, child):
        model_configs = []

        if len(child) > 0:
            for p in parent:
                for c in child:
                    combined_list = [p]
                    # Children at the lowest level of recursion will not be lists,
                    # so we need to check and handle this case correctly
                    combined_list.extend([c] if not isinstance(c, list) else c)

                    model_configs.append(combined_list)
        else:
            model_configs.extend(p for p in parent)

        return model_configs
