# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import os
from itertools import product

from .run_config import RunConfig
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig


class FullRunConfigGenerator:
    """
    A class that handles ModelAnalyzerConfig
    parsing and run_config generation
    """

    def __init__(self, analyzer_config, model_name):
        """
        analyzer_config : ModelAnalyzerConfig
            The config object parsed from 
            a model analyzer config yaml or json.

        model_name : str
            The name of the model for which we need
            to generate run configs
        """

        super().__init__()
        self._analyzer_config = analyzer_config
        self._model_name = model_name
        self._run_configs = []
        self._generate_all_run_configs()

    def __iter__(self):
        """
        Return an iterator over the data structure
        that holds RunConfigs
        """

        return iter(self._run_configs)

    def _generate_all_run_configs(self):
        """
        Returns
        -------
        list of dicts
            keys are parameters to perf_analyzer
            values are individual combinations of argument values
        """

        all_model_configs = self._generate_all_model_configs()
        all_perf_configs = self._generate_all_perf_configs()

        for model_config in all_model_configs:
            self._run_configs.append(RunConfig(model_config, all_perf_configs))

    def _generate_all_model_configs(self):
        """
        Generates all the possible ModelConfig objects
        """

        # TODO replace [None] with list from the config
        model_config_params = {
            'model_name': [self._model_name],
            'instance_group': [None],
            'dynamic_batching': [None]
        }

        model_configs = []
        for params in self._generate_parameter_combinations(
                model_config_params):
            model_path = os.path.join(self._analyzer_config.model_repository,
                                      params['model_name'])
            model_config = ModelConfig.create_from_file(model_path=model_path)

            # TODO edit model_config parameters
            model_configs.append(model_config)

        return model_configs

    def _generate_all_perf_configs(self):
        """
        Generates a list of PerfAnalyzerConfigs
        """

        perf_config_params = {
            'model-name': [self._model_name],
            'batch-size':
            self._analyzer_config.batch_sizes,
            'concurrency-range':
            self._analyzer_config.concurrency,
            'protocol': [self._analyzer_config.client_protocol],
            'url': [
                self._analyzer_config.triton_http_endpoint
                if self._analyzer_config.client_protocol == 'http' else
                self._analyzer_config.triton_grpc_endpoint
            ],
            'measurement-interval':
            [self._analyzer_config.perf_measurement_window],
        }

        perf_configs = []
        for params in self._generate_parameter_combinations(
                perf_config_params):
            perf_configs.append(PerfAnalyzerConfig(params))
        return perf_configs

    def _generate_parameter_combinations(self, params):
        """
        Generate a list of all possible subdictionaries
        from given dictionary. The subdictionaries will
        have all the same keys, but only one value from
        each key.

        Parameters
        ----------
        params : dict
            keys are strings and the values must be lists
        """

        param_combinations = list(product(*tuple(params.values())))
        return [dict(zip(params.keys(), vals)) for vals in param_combinations]
