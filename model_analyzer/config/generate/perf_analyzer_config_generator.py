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

#FIXME: Include this once ModelConfigGenerator refactor is checked in
#from model_analyzer.config.generate.generator_utils import GeneratorUtils

from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.config.input.config_defaults import DEFAULT_MEASUREMENT_MODE
from itertools import product


class PerfAnalyzerConfigGenerator:
    """ Given Perf Analyzer configuration options, generates Perf Analyzer configs """

    def __init__(self, cli_config, model_name):
        self._model_name = model_name

        self._batch_sizes = cli_config.batch_sizes
        self._concurrency = cli_config.concurrency

        # Extracts a dict of CLI config fields
        cli_config_fields = cli_config.get_all_config()

        self._perf_analyzer_flags = cli_config_fields['perf_analyzer_flags']
        self._client_protocol_is_http = (
            cli_config_fields['client_protocol'] == 'http')
        self._launch_mode_is_c_api = (
            cli_config_fields['triton_launch_mode'] == 'c_api')

        self._triton_install_path = cli_config_fields['triton_install_path']
        self._output_model_repo_path = cli_config_fields[
            'output_model_repository_path']
        self._protocol = cli_config_fields['client_protocol']
        self._triton_http_endpoint = cli_config_fields['triton_http_endpoint']
        self._triton_grpc_endpoint = cli_config_fields['triton_grpc_endpoint']
        self._client_protocol = cli_config_fields['client_protocol']

        self._generate_perf_config_fields()

    def is_done(self):
        """ Returns true if this generator is done generating configs """
        return len(self._configs) == 0

    def next_config(self):
        """ Returns the next generated config """
        return self._configs.pop(0)

    def _generate_perf_config_fields(self):
        perf_config_params = self._create_perf_config_params()

        self._configs = []

        for params in self._generate_parameter_combinations(perf_config_params):
            new_perf_config = PerfAnalyzerConfig()
            new_perf_config.update_config(params)
            # User provided flags can override the search parameters
            new_perf_config.update_config(self._perf_analyzer_flags)

            self._configs.append(new_perf_config)

    def _create_perf_config_params(self):
        perf_config_params = {
            'model-name': [self._model_name],
            'batch-size': self._batch_sizes,
            'concurrency-range': self._concurrency,
            'measurement-mode': [DEFAULT_MEASUREMENT_MODE]
        }

        if self._launch_mode_is_c_api:
            perf_config_params.update({
                'service-kind': ['triton_c_api'],
                'triton-server-directory': [self._triton_install_path],
                'model-repository': [self._output_model_repo_path]
            })
        else:
            perf_config_params.update({
                'protocol': [self._client_protocol],
                'url': [
                    self._triton_http_endpoint if self._client_protocol_is_http
                    else self._triton_grpc_endpoint
                ]
            })

        return perf_config_params

    # FIXME: Remove this once ModelConfigGenerator refactor is checked-in
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