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

from model_analyzer.config.input.objects.config_model_profile_spec \
    import ConfigModelProfileSpec
from model_analyzer.constants import THROUGHPUT_GAIN

import logging
import copy


class RunSearch:
    """
    A class responsible for searching the config space.
    """

    def __init__(self, config):
        self._max_concurrency = config.run_config_search_max_concurrency
        self._max_instance_count = config.run_config_search_max_instance_count
        self._max_preferred_batch_size = config.run_config_search_max_preferred_batch_size
        self._model_config_parameters = {'instance_count': 1}
        self._measurements = []
        self._last_batch_length = None

        # Run search operating mode
        self._sweep_mode_function = None

    def _create_model_config(self, cpu_only=False):
        """
        Generate the model config sweep to be used.
        """

        model_config = self._model_config_parameters

        new_config = {}
        if 'dynamic_batching' in model_config:
            if model_config['dynamic_batching'] is None:
                new_config['dynamic_batching'] = {}
            else:
                new_config['dynamic_batching'] = {
                    'preferred_batch_size': [model_config['dynamic_batching']]
                }

        if 'instance_count' in model_config:
            if not cpu_only:
                new_config['instance_group'] = [{
                    'count': model_config['instance_count'],
                    'kind': 'KIND_GPU'
                }]
            else:
                new_config['instance_group'] = [{
                    'count': model_config['instance_count'],
                    'kind': 'KIND_CPU'
                }]
        return new_config

    def add_measurements(self, measurements):
        """
        Add the measurments that are the result of running
        the sweeps.

        Parameters
        ----------
        measurements : list
            list of measurements
        """

        self._last_batch_length = len(measurements)

        # The list will contain one parameter, because we are experimenting
        # with one value at a time.
        self._measurements += measurements

    def _step_instance_count(self):
        """
        Advances instance count by one step.
        """

        self._model_config_parameters['instance_count'] += 1

    def _step_dynamic_batching(self):
        """
        Advances the dynamic batching by one step.
        """

        if 'dynamic_batching' not in self._model_config_parameters:
            # Enable dynamic batching
            self._model_config_parameters['dynamic_batching'] = None
        else:
            if self._model_config_parameters['dynamic_batching'] is None:
                self._model_config_parameters['dynamic_batching'] = 1
            else:
                self._model_config_parameters['dynamic_batching'] *= 2

    def _get_throughput(self, measurement):
        return measurement.get_metric('perf_throughput').value()

    def _calculate_throughput_gain(self, index):
        throughput_before = self._get_throughput(
            self._measurements[-(index + 1)])
        throughput_after = self._get_throughput(self._measurements[-index])
        gain = (throughput_after - throughput_before) / throughput_before
        return gain

    def _valid_throughput_gain(self):
        """
        Returns true if the amount of throughput gained
        is reasonable for continuing the search process
        """

        # If number of measurements is smaller than 4,
        # the search can continue.
        if len(self._measurements) < 4:
            return True

        return self._calculate_throughput_gain(1) > THROUGHPUT_GAIN or \
            self._calculate_throughput_gain(2) > THROUGHPUT_GAIN or \
            self._calculate_throughput_gain(3) > THROUGHPUT_GAIN

    def init_model_sweep(self, concurrency, search_model_config_parameters):
        """
        Intiliazes the sweep mode, and model config parameters in some cases.
        """

        # Reset the measurements after each init
        self._measurements = []
        if len(concurrency) != 0 and search_model_config_parameters:
            self._model_config_parameters = {'instance_count': 0}
            self._sweep_mode_function = self._sweep_model_config_only
        elif len(concurrency) == 0 and search_model_config_parameters:
            self._model_config_parameters = {'instance_count': 1}
            logging.info(
                'Will sweep both the concurrency and model config parameters...'
            )
            self._sweep_mode_function = self._sweep_concurrency_and_model_config
        else:
            logging.info('Will sweep only through the concurrency values...')
            self._sweep_mode_function = self._sweep_concurrency_only

    def get_model_sweep(self, config_model):
        """
        Get the next iteration of the sweeps.

        Parameters
        ----------
        config_model : ConfigModelProfileSpec
            The config model object of the model to sweep through

        Returns
        -------
        config_model, list
            The list may be empty, contain a model config dict or None
        """

        new_model = ConfigModelProfileSpec(
            copy.deepcopy(config_model.model_name()),
            copy.deepcopy(config_model.cpu_only()),
            copy.deepcopy(config_model.objectives()),
            copy.deepcopy(config_model.constraints()),
            copy.deepcopy(config_model.parameters()),
            copy.deepcopy(config_model.model_config_parameters()),
            copy.deepcopy(config_model.perf_analyzer_flags()))

        if self._sweep_mode_function:
            new_model, model_sweep = self._sweep_mode_function(new_model)

            # Only log message if there is new runs.
            if model_sweep:
                self._log_message(new_model)
            return new_model, model_sweep
        return new_model, []

    def _sweep_concurrency_and_model_config(self, model):
        """
        Gets next iteration of both the concurrency and model config
        parameters

        Parameters
        ----------
        model : ConfigModelProfileSpec
            The model whose parameters are being swept over
        """

        return self._sweep_parameters(model, sweep_model_configs=True)

    def _sweep_concurrency_only(self, model):
        """
        Gets next iteration of the concurrency sweep
        """

        return self._sweep_parameters(model, sweep_model_configs=False)

    def _sweep_parameters(self, model, sweep_model_configs):
        """
        A helper function that sweeps over concurrency
        and if required, over model configs as well
        """

        concurrency = model.parameters()['concurrency']

        if len(concurrency) == 0:
            model.parameters()['concurrency'] = [1]
        else:
            # Exponentially increase concurrency
            new_concurrency = concurrency[0] * 2

            # If the concurrency limit has been reached, the last batch lead to
            # an error, or the throughput gain is not significant, step
            # the concurrency value. TODO: add exponential backoff so
            # that the algorithm can step back and exactly find the points.
            concurrency_limit_reached = new_concurrency > self._max_concurrency
            last_batch_erroneous = self._last_batch_length == 0
            throughput_peaked = not self._valid_throughput_gain()
            if concurrency_limit_reached or last_batch_erroneous or throughput_peaked:

                # Reset concurrency
                if sweep_model_configs:
                    self._measurements = []
                    model.parameters()['concurrency'] = [1]

                    return self._sweep_model_config_only(model)
                else:
                    return model, []

            model.parameters()['concurrency'] = [new_concurrency]

        return model, [
            self._create_model_config(
                cpu_only=model.cpu_only()) if sweep_model_configs else None
        ]

    def _sweep_model_config_only(self, model):
        """
        Gets next iteration model config
        parameters sweep
        """

        self._step_instance_count()
        instance_limit_reached = self._model_config_parameters[
            'instance_count'] > self._max_instance_count

        if instance_limit_reached:
            # Reset instance_count
            self._model_config_parameters['instance_count'] = 1

            self._step_dynamic_batching()
            dynamic_batching_enabled = self._model_config_parameters[
                'dynamic_batching'] is not None

            if dynamic_batching_enabled:
                batch_size_limit_reached = self._model_config_parameters[
                    'dynamic_batching'] > self._max_preferred_batch_size
                if batch_size_limit_reached:
                    return model, []
        return model, [self._create_model_config(cpu_only=model.cpu_only())]

    def _log_message(self, model):
        """
        Writes the current state of the search to the console
        """

        concurrency = model.parameters()['concurrency'][0]
        message = 'dynamic batching is disabled.'
        if 'dynamic_batching' in self._model_config_parameters:
            if self._model_config_parameters['dynamic_batching'] is None:
                message = 'dynamic batching is enabled.'
            else:
                message = (
                    "preferred batch size is set to "
                    f"{self._model_config_parameters['dynamic_batching']}.")

        if self._sweep_mode_function == self._sweep_concurrency_only:
            logging.info(f"Concurrency set to {concurrency}. ")
        elif self._sweep_mode_function == self._sweep_concurrency_and_model_config:
            logging.info(
                f"Concurrency set to {concurrency}. "
                f"Instance count set to "
                f"{self._model_config_parameters['instance_count']}, and {message}"
            )
        elif self._sweep_mode_function == self._sweep_model_config_only:
            logging.info(
                f"Instance count set to "
                f"{self._model_config_parameters['instance_count']}, and {message}"
            )
