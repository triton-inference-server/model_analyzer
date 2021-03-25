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

from itertools import product
from sys import flags

from .run_config import RunConfig
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig


class RunConfigGenerator:
    """
    A class that handles ModelAnalyzerConfig parsing, generation, and
    exection of a list of run configurations.
    """
    def __init__(self, config, client):
        """
        Parameters
        ----------
        config : ModelAnalyzerConfig
            The config object parsed from
            a model analyzer config yaml or json.
        client : TritonClient
            TritonClient to be used for interacting with the Triton API
        """

        self._analyzer_config = config.get_all_config()
        self._run_configs = []
        self._client = client
        self._model_name_index = 0
        self._model_configs = []

    def run_configs(self):
        """
        Returns
        -------
        list
            The run configs currently in this generator
        """

        return self._run_configs

    def next_config(self):
        """
        Returns list of run configs
        """

        return self._run_configs.pop()

    def clear_configs(self):
        """
        Empties the list of run_configs
        """

        self._run_configs = []

    def generate_run_config_for_model_sweep(self, model, model_sweep):
        """
        Parameters
        ----------
        model : ConfigModel
            The model for which a run config is being generated
        model_sweep: dict
            Model config parameters
        """

        analyzer_config = self._analyzer_config
        model_repository = analyzer_config['model_repository']
        num_retries = analyzer_config['max_retries']

        if analyzer_config['triton_launch_mode'] != 'remote':
            model_config = ModelConfig.create_from_file(
                f'{model_repository}/{model.model_name()}')

            if model_sweep is not None:
                model_config_dict = model_config.get_config()
                for key, value in model_sweep.items():
                    if value is not None:
                        model_config_dict[key] = value
                model_config = ModelConfig.create_from_dictionary(
                    model_config_dict)

            model_name_index = self._model_name_index
            model_config_dict = model_config.get_config()

            try:
                model_name_index = self._model_configs.index(model_config_dict)
            except ValueError:
                self._model_configs.append(model_config_dict)
                self._model_name_index += 1

            # Temporary model name to be used for profiling. We
            # can't use the same name for different configurations.
            # The new model name is the original model suffixed with
            # _i<config_index>. Where the config index is the index
            # of the model config alternative.
            model_tmp_name = f'{model.model_name()}_i{model_name_index}'
            model_config.set_field('name', model_tmp_name)
            perf_configs = self._generate_perf_config_for_model(
                model_tmp_name, model)
            for perf_config in perf_configs:
                self._run_configs.append(
                    RunConfig(model.model_name(), model_config, perf_config))
        else:
            model_config = ModelConfig.create_from_triton_api(
                self._client, model.model_name(), num_retries)
            perf_configs = self._generate_perf_config_for_model(
                model.model_name(), model)

            for perf_config in perf_configs:
                # Add the new run config.
                self._run_configs.append(
                    RunConfig(model.model_name(), model_config, perf_config))

    def generate_model_config_combinations(self, value):
        """
        Generates all the alternative config fields for
        a given value.

        Parameters
        ----------
        value : object
            The value to be used for sweeping.

        Returns
        -------
        list
            A list of all the alternatives for the model config
            parameters.
        """

        if type(value) is dict:
            sweeped_dict = {}
            for key, sweep_choices in value.items():
                sweep_parameter_list = []

                # This is the list of sweep parameters. When parsing the Model
                # Analyzer Config every sweepable parameter will be converted
                # to a list of values to make the parameter sweeping easier in
                # here.
                for sweep_choice in sweep_choices:
                    sweep_parameter_list += \
                        self.generate_model_config_combinations(
                                                    sweep_choice
                                                    )

                sweeped_dict[key] = sweep_parameter_list

            # Generate parameter combinations for this field.
            return self._generate_parameter_combinations(sweeped_dict)

        # When this line of code is executed the value for this field is
        # a list. This list does NOT represent possible sweep values.
        # Because of this we need to ensure that in every sweep configuration,
        # one item from every list item exists.
        elif type(value) is list:

            # This list contains a set of lists. The return value from this
            # branch of the code is a list of lists where in each inner list
            # there is one item from every list item.
            sweep_parameter_list = []
            for item in value:
                sweep_parameter_list_item = \
                    self.generate_model_config_combinations(
                        item)
                sweep_parameter_list.append(sweep_parameter_list_item)

            # Cartesian product of all the elements in the sweep_parameter_list
            return [list(x) for x in list(product(*sweep_parameter_list))]

        # In the default case return a list of the value. This function should
        # always return a list.
        return [value]

    def _generate_perf_config_for_model(self, model_name, config_model):
        """
        Generates a list of PerfAnalyzerConfigs
        """

        perf_config_params = {
            'model-name': [model_name],
            'batch-size':
            config_model.parameters()['batch_sizes'],
            'concurrency-range':
            config_model.parameters()['concurrency'],
            'protocol': [self._analyzer_config['client_protocol']],
            'url': [
                self._analyzer_config['triton_http_endpoint']
                if self._analyzer_config['client_protocol'] == 'http' else
                self._analyzer_config['triton_grpc_endpoint']
            ],
            'measurement-interval':
            [self._analyzer_config['perf_measurement_window']],
        }

        perf_configs = []
        for params in self._generate_parameter_combinations(
                perf_config_params):
            perf_config = PerfAnalyzerConfig()
            perf_config.update_config(config_model.perf_analyzer_flags())
            perf_config.update_config(params)
            perf_configs.append(perf_config)
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
