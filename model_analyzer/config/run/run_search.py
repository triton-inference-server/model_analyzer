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

from model_analyzer.config.input.config_model import ConfigModel

MAX_CONCURRENCY = 256
MAX_DYNAMIC_BATCH_SIZE = 16
MAX_INSTANCE_COUNT = 5


class RunSearch:
    """A class responsible for searching the config space.
    """
    def __init__(self):
        self._sweeps = []
        self._model_config = {'instance_count': 1}
        self._checked = False
        self._measurements = []
        self._last_batch_length = None

    def _create_model_config(self):
        """
        Generate the model config sweep to be used.
        """
        model_config = self._model_config

        new_config = {}
        if 'dynamic_batching' in model_config:
            if model_config['dynamic_batching'] is None:
                new_config['dynamic_batching'] = {}
            else:
                new_config['dynamic_batching'] = {
                    'preferred_batch_size': [model_config['dynamic_batching']]
                }

        if 'instance_count' in model_config:
            new_config['instance_group'] = [{
                'count':
                model_config['instance_count'],
                'kind':
                'KIND_GPU'
            }]
        return new_config

    def add_run_results(self, measurements):
        """
        Add the measurments that are the result of running
        the sweeps.

        Parameters
        ----------
        measurements : dict
            A dictionary where the keys are the Model Configs and
            the values are measurements.
        """

        self._last_batch_length = len(list(measurements.values()))
        self._measurements += list(measurements.values())

    def _step_instance_count(self):
        """
        Advances instance count by one step.
        """

        self._model_config['instance_count'] += 1

    def _step_dynamic_batching(self):
        """
        Advances the dynamic batching by one step.
        """

        if 'dynamic_batching' not in self._model_config:
            # Enable dynamic batching
            self._model_config['dynamic_batching'] = None
        else:
            if self._model_config['dynamic_batching'] is None:
                self._model_config['dynamic_batching'] = 1
            else:
                self._model_config['dynamic_batching'] *= 2

    def _get_throughput(self, measurement):
        return measurement.get_value_of_metric('perf_throughput').value()

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

        if self._calculate_throughput_gain(1) > 0.05 or \
                self._calculate_throughput_gain(2) > 0.05 or \
                self._calculate_throughput_gain(3) > 0.05:
            return True
        else:
            return False

    def get_model_sweeps(self,
                         config_model,
                         model_sweeps,
                         search_model_config_parameters=True):
        """
        Get the next iteration of the sweeps.

        Parameters
        ----------
        config_model : ConfigModel
            The config model object of the model to sweep through
        model_sweeps : list
            If the model already has a list of config sweeps
            this parameter will contain the config sweeps.
            Otherwise, the array should be empty.
        search_model_config_parameters : bool
            Whether to automatically search of the model config parameters
            or only search for concurrency values.
        """
        tmp_model = ConfigModel(config_model.model_name(),
                                config_model.objectives(),
                                config_model.constraints(),
                                config_model.parameters(),
                                config_model.model_config_parameters())

        concurrency = tmp_model.parameters()['concurrency']

        if not self._checked:
            self._checked = True
            if len(concurrency) != 0 and \
                    search_model_config_parameters is False:
                return config_model, model_sweeps

        if len(concurrency) == 0:
            concurrency = 1
            tmp_model.parameters()['concurrency'] = [1]
        else:
            # Exponentially increase concurrency
            new_concurrency = concurrency[0] * 2
            tmp_model.parameters()['concurrency'] = [new_concurrency]

            # If the concurrency limit has been reached, the last batch lead to
            # an error, or the throughput gain is not significant, step
            # advancing the concurrency value. TODO: add exponential backoff so
            # that the algorithm can step back and exactly find the points.
            if new_concurrency > MAX_CONCURRENCY or self._last_batch_length == 0 or not self._valid_throughput_gain():
                # Reset concurrency
                self._measurements = []
                tmp_model.parameters()['concurrency'] = [1]
                self.step_instance_count()
                if self._model_config['instance_count'] == MAX_INSTANCE_COUNT:
                    # Reset instance_count
                    self._model_config['instance_count'] = 1
                    self.step_dynamic_batching()
                    if self._model_config[
                            'dynamic_batching'] == MAX_DYNAMIC_BATCH_SIZE:
                        return tmp_model, []

        return tmp_model, [self._create_model_config()]
