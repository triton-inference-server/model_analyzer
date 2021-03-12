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
import os
import logging
from model_analyzer.output.file_writer import FileWriter

from .run_config import RunConfig
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.result.measurement import Measurement


class RunConfigGenerator:
    """
    A class that handles ModelAnalyzerConfig parsing, generation, and
    exection of a list of run configurations.
    """

    def __init__(self,
                 model,
                 analyzer_config,
                 client,
                 server,
                 result_manager,
                 metrics_manager,
                 run_search=None,
                 generate_only=False):
        """
        analyzer_config : ModelAnalyzerConfig
            The config object parsed from
            a model analyzer config yaml or json.

        model : ConfigModel
            ConfigModel object to generate the run configs for.

        client : TritonClient
            TritonClient to be used for interacting with the Triton API
        """

        self._analyzer_config = analyzer_config.get_all_config()
        self._model = model
        self._run_configs = []
        self._run_search = run_search
        self._client = client
        self._server = server
        self._run_search = run_search
        self._metrics_manager = metrics_manager
        self._result_manager = result_manager
        self._model_name_index = 0
        self._model_configs = []
        self._generate_only = generate_only
        self._generate_run_configs()

    def _generate_model_config_combinations(self, value):
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
                        self._generate_model_config_combinations(
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
                    self._generate_model_config_combinations(
                        item)
                sweep_parameter_list.append(sweep_parameter_list_item)

            # Cartesian product of all the elements in the sweep_parameter_list
            return [list(x) for x in list(product(*sweep_parameter_list))]

        # In the default case return a list of the value. This function should
        # always return a list.
        return [value]

    def execute_run_configs(self):

        max_retries = self._analyzer_config['max_retries']
        output_model_repo_path = self._analyzer_config[
            'output_model_repository_path']
        model_repository = self._analyzer_config['model_repository']
        perf_output = self._analyzer_config['perf_output']
        triton_launch_mode = self._analyzer_config['triton_launch_mode']

        measurements = {}
        while self._run_configs:
            # Remove one run config from the list
            run_config = self._run_configs.pop()

            model_config = run_config.model_config()
            original_model_name = run_config.model_name()
            perf_config = run_config.perf_config()
            model_config_name = model_config.get_field('name')
            measurements[model_config] = []

            # If the model config already exists, do not recreate the
            # directory.
            if not os.path.exists(
                    f'{output_model_repo_path}/{model_config_name}'
            ) and triton_launch_mode != 'remote':
                # Create the directory for the new model
                os.mkdir(f'{output_model_repo_path}/{model_config_name}')
                model_config.write_config_to_file(
                    f'{output_model_repo_path}/{model_config_name}', True,
                    f'{model_repository}/{original_model_name}')

            self._server.start()
            self._client.wait_for_server_ready(max_retries)
            status = self._client.load_model(model_name=model_config_name)
            if status == -1:
                self._server.stop()
                continue

            status = self._client.wait_for_model_ready(
                model_name=model_config_name, num_retries=max_retries)
            if status == -1:
                self._server.stop()
                continue

            # Profile various batch size and concurrency values.
            # TODO: Need to sort the values for batch size and concurrency
            # for correct measurment of the GPU memory metrics.
            perf_output_writer = None if \
                not perf_output else FileWriter()

            logging.info(f"Profiling model {perf_config['model-name']}...")
            gpu_data, non_gpu_data = self._metrics_manager.profile_model(
                perf_config=perf_config, perf_output_writer=perf_output_writer)
            if gpu_data is not None and non_gpu_data is not None:
                measurement = Measurement(gpu_data=gpu_data,
                                          non_gpu_data=non_gpu_data,
                                          perf_config=perf_config)
                self._result_manager.add_measurement(run_config, measurement)
                measurements[model_config].append(measurement)

            self._server.stop()
        return measurements

    def _generate_run_config_for_model_sweep(self, model, model_sweep):
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

    def _generate_run_configs(self):
        model = self._model
        analyzer_config = self._analyzer_config
        triton_launch_mode = analyzer_config['triton_launch_mode']

        model_config_parameters = model.model_config_parameters()

        if analyzer_config['run_config_search_disable']:
            model_sweeps = \
                self._generate_model_config_combinations(
                    model_config_parameters)
            if triton_launch_mode != 'remote':
                for model_sweep in model_sweeps:
                    self._generate_run_config_for_model_sweep(
                        model, model_sweep)
            else:
                self._generate_run_config_for_model_sweep(model, None)

            if not self._generate_only:
                measurements = self.execute_run_configs()
            return

        if triton_launch_mode == 'remote':
            search_model_config = False
            model = self._run_search.init_model_sweep(model,
                                                      search_model_config)
            recurring_model_sweeps = []
        else:
            if model_config_parameters is not None:
                recurring_model_sweeps = \
                    self._generate_model_config_combinations(
                        model_config_parameters)
                search_model_config = False
                model = self._run_search.init_model_sweep(
                    model, search_model_config)
            else:
                recurring_model_sweeps = []
                search_model_config = True
                model = self._run_search.init_model_sweep(
                    model, search_model_config)

        has_run_once = False
        if len(recurring_model_sweeps) == 0:
            model_sweeps = []
            model, new_model_sweeps = \
                self._run_search.get_model_sweeps(model)
            model_sweeps += new_model_sweeps

            while model_sweeps:
                for model_sweep in model_sweeps:
                    has_run_once = True
                    self._generate_run_config_for_model_sweep(
                        model, model_sweep)

                # Empty the model_sweeps after they are added to the
                # list
                model_sweeps = []

                if not self._generate_only:
                    measurements = self.execute_run_configs()
                    self._run_search.add_run_results(measurements)
                    model, new_model_sweeps = \
                        self._run_search.get_model_sweeps(model)
                    model_sweeps += new_model_sweeps

        else:
            for recurring_model_sweep in recurring_model_sweeps:
                model = self._model
                model = self._run_search.init_model_sweep(
                    model, search_model_config)
                model, model_sweeps = \
                    self._run_search.get_model_sweeps(model)

                while model_sweeps:
                    has_run_once = True
                    self._generate_run_config_for_model_sweep(
                        model, recurring_model_sweep)
                    model_sweeps = []

                    if not self._generate_only:
                        measurements = self.execute_run_configs()
                        self._run_search.add_run_results(measurements)
                        model, new_model_sweeps = \
                            self._run_search.get_model_sweeps(model)
                        model_sweeps += new_model_sweeps

        if not has_run_once:
            has_run_once = True
            for model_sweep in recurring_model_sweeps:
                self._generate_run_config_for_model_sweep(model, model_sweep)

            if len(recurring_model_sweeps) == 0:
                self._generate_run_config_for_model_sweep(model, None)

            if not self._generate_only:
                measurements = self.execute_run_configs()
            return

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
