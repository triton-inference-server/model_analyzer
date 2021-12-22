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
from model_analyzer.config.run.run_search import RunSearch
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
import logging

logger = logging.getLogger(LOGGER_NAME)


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

        self._config = config
        self._analyzer_config = config.get_all_config()
        self._run_search = RunSearch(config=config)
        self._run_configs = []
        self._client = client
        self._model_name_index = 0
        self._model_configs = []

    def init(self, model):

        if self._config.run_config_search_disable:
            logger.info(
                f"Running manual config search for model: {model.model_name()}")
            self._run_model_no_search(model)
        else:
            logger.info(
                f"Running auto config search for model: {model.model_name()}")
            self._run_model_with_search(model)

    def is_done(self):
        """ Return true if all RunConfigs for the model have been returned """
        return len(self._run_configs) == 0

    def add_measurement(self, measurement):
        """ 
        Given the results from the last RunConfig, make decisions 
        about future configurations to generate
        """
        if measurement:
            self._run_search.add_measurements([measurement])

    def _run_model_no_search(self, model):
        """
        Creates run configs from specified combinations and executes
        them without any run search
        """

        # Generate all the run configs at once and return
        if self._config.triton_launch_mode != 'remote':
            user_model_config_sweeps = \
                self.generate_model_config_combinations(
                    model.model_config_parameters())
            for user_model_config_sweep in user_model_config_sweeps:
                self.generate_run_config_for_model_sweep(
                    model, user_model_config_sweep)
        else:
            self.generate_run_config_for_model_sweep(model, None)

    def _run_model_with_search(self, model):
        """
        Searches over the required config elements,
        creates run configs and executes them
        """

        model_config_parameters = model.model_config_parameters()

        # Run config search is enabled, figure out which parameters to sweep over and do sweep
        if self._config.triton_launch_mode == 'remote':
            self._run_model_config_sweep(model, search_model_config=False)
        else:
            user_model_config_sweeps = \
                self.generate_model_config_combinations(
                    model_config_parameters)
            if model.parameters()['concurrency']:
                # Both are specified, search over neither
                for user_model_config_sweep in user_model_config_sweeps:
                    self.generate_run_config_for_model_sweep(
                        model, user_model_config_sweep)
            else:
                # Search through concurrency values only
                for user_model_config_sweep in user_model_config_sweeps:
                    self._run_model_config_sweep(
                        model,
                        search_model_config=False,
                        user_model_config_sweep=user_model_config_sweep)

            # If no model config parameters were specified, then we only ran the default
            # configuration above and need to do an automatic sweep
            #
            if not model_config_parameters:
                self._run_model_config_sweep(model, search_model_config=True)

    def _run_model_config_sweep(self,
                                model,
                                search_model_config,
                                user_model_config_sweep=None):
        """
        Initializes the model sweep, iterates until search bounds,
        and executes run configs
        """

        self._run_search.init_model_sweep(model.parameters()['concurrency'],
                                          search_model_config)

        next_model = model

        while True:
            # Get next model sweep
            next_model, auto_model_config_sweep = self._run_search.get_next_model_sweep(
                next_model)

            # End search when get_model sweep returns empty
            if not auto_model_config_sweep:
                return
            if user_model_config_sweep:
                model_sweep_for_run_config = user_model_config_sweep
            else:
                model_sweep_for_run_config = auto_model_config_sweep[0]

            self.generate_run_config_for_model_sweep(
                next_model, model_sweep_for_run_config)

    def next_config(self):
        """
        Returns
        -------
        RunConfig
            The next run config in the run config list
            generated by this instance
        """

        return self._run_configs.pop(0)

    def clear_configs(self):
        """
        Empties the list of run_configs
        """

        self._run_configs = []
        self._model_configs = []
        self._model_name_index = 0

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
        num_retries = analyzer_config['client_max_retries']

        if analyzer_config['triton_launch_mode'] == 'remote':
            reload_model = not analyzer_config['reload_model_disable']
            if reload_model:
                self._client.load_model(model.model_name())
            model_config = ModelConfig.create_from_triton_api(
                self._client, model.model_name(), num_retries)
            if reload_model:
                self._client.unload_model(model.model_name())
            model_config.set_cpu_only(model.cpu_only())
            perf_configs = self._generate_perf_config_for_model(
                model.model_name(), model,
                analyzer_config['triton_launch_mode'])

            for perf_config in perf_configs:
                # Add the new run config.
                self._run_configs.append(
                    RunConfig(model.model_name(), model_config, perf_config,
                              None))
        else:
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
            model_config.set_cpu_only(model.cpu_only())
            perf_configs = self._generate_perf_config_for_model(
                model_tmp_name, model, analyzer_config['triton_launch_mode'])
            for perf_config in perf_configs:
                self._run_configs.append(
                    RunConfig(model.model_name(), model_config, perf_config,
                              model.triton_server_environment()))

    def generate_model_config_combinations(self, value):
        configs = self._generate_model_config_combinations_helper(value)
        if not self._is_default_config_in_configs(configs):
            self._add_default_config(configs)
        return configs

    def _is_default_config_in_configs(self, configs):
        return None in configs

    def _add_default_config(self, configs):
        # Add in an empty configuration, which will apply the default values
        configs.append(None)

    def _generate_model_config_combinations_helper(self, value):
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
                        self._generate_model_config_combinations_helper(
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
                    self._generate_model_config_combinations_helper(
                        item)
                sweep_parameter_list.append(sweep_parameter_list_item)

            # Cartesian product of all the elements in the sweep_parameter_list
            return [list(x) for x in list(product(*sweep_parameter_list))]

        # In the default case return a list of the value. This function should
        # always return a list.
        return [value]

    def _generate_perf_config_for_model(self, model_name, config_model,
                                        launch_mode):
        """
        Generates a list of PerfAnalyzerConfigs based on
        the config_mode and launch_mode
        """

        perf_config_params = {
            'model-name': [model_name],
            'batch-size': config_model.parameters()['batch_sizes'],
            'concurrency-range': config_model.parameters()['concurrency'],
            'measurement-mode': ['count_windows']
        }

        if launch_mode == 'c_api':
            perf_config_params.update({
                'service-kind': ['triton_c_api'],
                'triton-server-directory': [
                    self._analyzer_config['triton_install_path']
                ],
                'model-repository': [
                    self._analyzer_config['output_model_repository_path']
                ]
            })
        else:
            perf_config_params.update({
                'protocol': [self._analyzer_config['client_protocol']],
                'url': [
                    self._analyzer_config['triton_http_endpoint']
                    if self._analyzer_config['client_protocol'] == 'http' else
                    self._analyzer_config['triton_grpc_endpoint']
                ]
            })

        perf_configs = []
        for params in self._generate_parameter_combinations(perf_config_params):
            perf_config = PerfAnalyzerConfig()
            perf_config.update_config(params)
            # User provided flags can override the search parameters
            perf_config.update_config(config_model.perf_analyzer_flags())
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
