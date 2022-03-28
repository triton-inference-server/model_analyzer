# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.result.results import Results
from model_analyzer.result.measurement import Measurement
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig

import unittest
from unittest.mock import MagicMock, patch
from .common import test_result_collector as trc


class TestResults(trc.TestResultCollector):

    def setUp(self):
        self._construct_results()

    def tearDown(self):
        NotImplemented

    def test_contains_model(self):
        """
        Test for the existence of expected models
        """
        self.assertTrue(self._result.contains_model('modelA'))
        self.assertTrue(self._result.contains_model('modelB'))
        self.assertFalse(self._result.contains_model('modelC'))

    def test_contains_model_config(self):
        """
        Test for the existence of expected model configs in modelA
        """
        self.assertTrue(
            self._result.contains_model_config('modelA', 'model_config_0'))
        self.assertTrue(
            self._result.contains_model_config('modelA', 'model_config_1'))
        self.assertTrue(
            self._result.contains_model_config('modelA', 'model_config_2'))
        self.assertFalse(
            self._result.contains_model_config('modelA', 'model_config_3'))
        self.assertFalse(
            self._result.contains_model_config('modelC', 'model_config_0'))

    def test_get_list_of_models(self):
        """
        Test that the list of models is returned correctly
        """
        model_list = self._result.get_list_of_models()

        self.assertEqual(model_list, ['modelA', 'modelB'])

    def test_get_list_of_model_config_measurements(self):
        """
        Test that the correct number of measurements is returned per model config
        """
        model_config_measurements_list = self._result.get_list_of_model_config_measurements(
        )
        num_model_config_measurements = [
            len(mcm) for mcm in model_config_measurements_list
        ]

        self.assertEqual(num_model_config_measurements[0], 3)
        self.assertEqual(num_model_config_measurements[1], 2)

    def test_get_measurements_dict(self):
        """
        Test that the measurements were correctly added to the dictionary
        """
        measurements = self._result.get_list_of_measurements()

        self.assertEqual(measurements, self._measurements_added)

    def test_get_model_measurements_dict(self):
        """
        Test that the measurements were correctly added to the model dictionairies
        """
        model_measurements = self._result.get_model_measurements_dict('modelA')

        for (index, (model_config,
                     measurements)) in enumerate(model_measurements.values()):
            self.assertEqual(model_config,
                             self._model_run_config[index].model_config())
            self.assertEqual(measurements, self._measurements[index])

    def test_get_model_config_measurements_dict(self):
        """
        Test that the measurements were correctly added to the model config dictionaries
        """
        model_config_measurements = self._result.get_model_config_measurements_dict(
            'modelA', 'model_config_1')

        self.assertEqual(model_config_measurements, self._measurements[1])

    def test_get_all_model_config_measurements(self):
        """
        Test that the list of measurements were correctly generated for a 
        given model + model_config combination
        """
        model_config, measurements = self._result.get_all_model_config_measurements(
            'modelA', 'model_config_1')

        self.assertEqual(model_config, self._model_run_config[1].model_config())
        self.assertEqual(measurements, list(self._measurements[1].values()))

    def _construct_results(self):
        """
        Creates an instance of the results class with two models, and 
        various model configs. Treating the measurements as strings,
        which makes debugging easier
        """
        self._result = Results()

        self._model_run_config = []
        self._model_run_config.append(
            self._create_model_run_config('modelA', 'model_config_0'))

        self._model_run_config.append(
            self._create_model_run_config('modelA', 'model_config_1'))

        self._model_run_config.append(
            self._create_model_run_config('modelA', 'model_config_2'))

        self._measurements = []
        self._measurements.append({"key_A": "1", "key_B": "2", "key_C": "3"})
        self._measurements.append({"key_D": "4", "key_E": "5", "key_F": "6"})
        self._measurements.append({"key_G": "7", "key_H": "8", "key_I": "9"})

        self._result.add_measurement(self._model_run_config[0], "key_A", "1")
        self._result.add_measurement(self._model_run_config[0], "key_B", "2")
        self._result.add_measurement(self._model_run_config[0], "key_C", "3")

        self._result.add_measurement(self._model_run_config[1], "key_D", "4")
        self._result.add_measurement(self._model_run_config[1], "key_E", "5")
        self._result.add_measurement(self._model_run_config[1], "key_F", "6")

        self._result.add_measurement(self._model_run_config[2], "key_G", "7")
        self._result.add_measurement(self._model_run_config[2], "key_H", "8")
        self._result.add_measurement(self._model_run_config[2], "key_I", "9")

        model_run_config_0 = self._create_model_run_config(
            'modelB', 'model_config_0')
        self._result.add_measurement(model_run_config_0, "key_F", "6")
        self._result.add_measurement(model_run_config_0, "key_E", "5")
        self._result.add_measurement(model_run_config_0, "key_D", "4")

        model_run_config_1 = self._create_model_run_config(
            'modelB', 'model_config_1')
        self._result.add_measurement(model_run_config_1, "key_C", "3")
        self._result.add_measurement(model_run_config_1, "key_B", "2")
        self._result.add_measurement(model_run_config_1, "key_A", "1")

        self._measurements_added = [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '6', '5', '4', '3',
            '2', '1'
        ]

    def _create_model_run_config(self, model_name, model_config_name):
        model_config_dict = {'name': model_config_name}
        self._model_config = ModelConfig.create_from_dictionary(
            model_config_dict)

        return ModelRunConfig(model_name, self._model_config, None)


if __name__ == '__main__':
    unittest.main()
