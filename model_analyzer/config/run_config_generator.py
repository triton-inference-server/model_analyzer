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

from .run_config import RunConfig
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig


class RunConfigGenerator:
    """
    A class that handles ModelAnalyzerConfig parsing and generates a list of
    run configurations.
    """
    def __init__(self, model, analyzer_config):
        """
        analyzer_config : ModelAnalyzerConfig
            The config object parsed from
            a model analyzer config yaml or json.

        model_name : str
            The name of the model for which we need
            to generate run configs
        """

        super().__init__()
        self._analyzer_config = analyzer_config.get_all_config()
        self._model = model
        self._run_configs = []
        self._generate_run_configs()

    def _generate_run_config_for_field(self, value):
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

        if value is None:
            return [{}]

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
                        self._generate_run_config_for_field(
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
                    self._generate_run_config_for_field(
                        item)
                sweep_parameter_list.append(sweep_parameter_list_item)

            # Cartesian product of all the elements in the sweep_parameter_list
            return [list(x) for x in list(product(*sweep_parameter_list))]

        # In the default case return a list of the value. This function should
        # always return a list.
        return [value]

    def _generate_run_configs(self):
        analyzer_config = self._analyzer_config
        model_repository = analyzer_config['model_repository']
        model = self._model

        model_name_index = 0
        model_config_parameters = model.model_config_parameters()

        # Generate all the sweeps for a given parameter
        models_sweeps = \
            self._generate_run_config_for_field(
                model_config_parameters)
        for model_sweep in models_sweeps:
            model_config = ModelConfig.create_from_file(
                f'{model_repository}/{model.model_name()}')
            model_config_dict = model_config.get_config()
            for key, value in model_sweep.items():
                model_config_dict[key] = value
            model_config = ModelConfig.create_from_dictionary(
                model_config_dict)

            # Temporary model name to be used for profiling. We
            # can't use the same name for different configurations.
            # The new model name is the original model suffixed with
            # _i<config_index>. Where the config index is the index
            # of the model config alternative.
            model_tmp_name = f'{model.model_name()}_i{model_name_index}'
            model_config.set_field('name', model_tmp_name)
            perf_configs = self._generate_perf_config_for_model(
                model_tmp_name, model)

            # Add the new run config.
            self._run_configs.append(
                RunConfig(model.model_name(), model_config, perf_configs))
            model_name_index += 1

    def get_run_configs(self):
        """
        Returns list of run configs
        """

        return self._run_configs

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
            perf_configs.append(PerfAnalyzerConfig(config_model, params))
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
