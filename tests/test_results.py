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
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.config.run.run_config import RunConfig

import unittest
from unittest.mock import MagicMock, patch
from .common import test_result_collector as trc


class TestResults(trc.TestResultCollector):

    def setUp(self):
        self._construct_results()

    def tearDown(self):
        NotImplemented

    def _construct_results(self):
        self._result = Results()

        self._run_config = []
        self._run_config.append(
            self._create_run_config('modelA', 'model_config_0'))

        self._run_config.append(
            self._create_run_config('modelA', 'model_config_1'))

        self._run_config.append(
            self._create_run_config('modelA', 'model_config_2'))

        self._measurements = []
        self._measurements.append({"key_A": "1", "key_B": "2", "key_C": "3"})
        self._measurements.append({"key_D": "4", "key_E": "5", "key_F": "6"})
        self._measurements.append({"key_G": "7", "key_H": "8", "key_I": "9"})

        self._result.add_measurement(self._run_config[0], "key_A", "1")
        self._result.add_measurement(self._run_config[0], "key_B", "2")
        self._result.add_measurement(self._run_config[0], "key_C", "3")

        self._result.add_measurement(self._run_config[1], "key_D", "4")
        self._result.add_measurement(self._run_config[1], "key_E", "5")
        self._result.add_measurement(self._run_config[1], "key_F", "6")

        self._result.add_measurement(self._run_config[2], "key_G", "7")
        self._result.add_measurement(self._run_config[2], "key_H", "8")
        self._result.add_measurement(self._run_config[2], "key_I", "9")

        run_config_0 = self._create_run_config('modelB', 'model_config_0')
        self._result.add_measurement(run_config_0, "key_F", "6")
        self._result.add_measurement(run_config_0, "key_E", "5")
        self._result.add_measurement(run_config_0, "key_D", "4")

        run_config_1 = self._create_run_config('modelB', 'model_config_1')
        self._result.add_measurement(run_config_1, "key_C", "3")
        self._result.add_measurement(run_config_1, "key_B", "2")
        self._result.add_measurement(run_config_1, "key_A", "1")

    def test_contains_model(self):
        self.assertTrue(self._result.contains_model('modelA'))
        self.assertTrue(self._result.contains_model('modelB'))
        self.assertFalse(self._result.contains_model('modelC'))

    def test_contains_model_config(self):
        self.assertTrue(
            self._result.contains_model_config('modelA', 'model_config_0'))
        self.assertTrue(
            self._result.contains_model_config('modelA', 'model_config_1'))
        self.assertTrue(
            self._result.contains_model_config('modelA', 'model_config_2'))
        self.assertFalse(
            self._result.contains_model_config('modelA', 'model_config_3'))

    def test_is_done(self):
        resultA = self._result.next_result('modelA')
        resultB = self._result.next_result('modelB')

        modelA_config_cnt = 0
        while not self._result.is_done('modelA'):
            next(resultA)
            modelA_config_cnt = modelA_config_cnt + 1

        modelB_config_cnt = 0
        while not self._result.is_done('modelB'):
            next(resultB)
            modelB_config_cnt = modelB_config_cnt + 1

        self.assertEqual(modelA_config_cnt, 3)
        self.assertEqual(modelB_config_cnt, 2)

    def test_next_result(self):
        result = self._result.next_result('modelA')

        for index, run_config in enumerate(self._run_config):
            model_config, measurements = next(result)

            self.assertEqual(model_config, run_config.model_configs()[0])
            self.assertEqual(measurements, self._measurements[index])

    def test_get_all_model_config_measurements(self):
        model_config, measurements = self._result.get_all_model_config_measurements(
            'modelA', 'model_config_1')

        self.assertEqual(model_config, self._run_config[1].model_configs()[0])
        self.assertEqual(measurements, list(self._measurements[1].values()))

    def _create_run_config(self, model_name, model_config_name):
        model_config_dict = {'name': model_config_name}
        self._model_config = ModelConfig.create_from_dictionary(
            model_config_dict)

        return RunConfig(model_name, [self._model_config], None, None)

    def test_from_dict(self):
        result_dict = self._result.__dict__
        result_from_dict = Results.from_dict(result_dict)

        self.assertEqual(
            result_from_dict.get_all_model_config_measurements(
                'modelA', 'model_config_1'),
            self._result.get_all_model_config_measurements(
                'modelA', 'model_config_1'))


if __name__ == '__main__':
    unittest.main()
