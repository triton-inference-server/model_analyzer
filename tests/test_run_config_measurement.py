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

from tests.common.test_utils import convert_non_gpu_metrics_to_data, \
    convert_gpu_metrics_to_data, convert_avg_gpu_metrics_to_data, \
    construct_perf_analyzer_config, construct_run_config_measurement, default_encode

from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.result.model_config_measurement import ModelConfigMeasurement
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

from statistics import mean

import unittest
from unittest.mock import MagicMock, patch
from .common import test_result_collector as trc

import json


class TestRunConfigMeasurement(trc.TestResultCollector):

    def setUp(self):
        self._construct_rcm0()
        self._construct_rcm1()

    def tearDown(self):
        NotImplemented

    def test_model_variants_name(self):
        """
        Test that the model_variants_name was initialized correctly
        """
        self.assertEqual(self.rcm0.model_variants_name(),
                         self.model_variants_name)

    def test_gpu_data(self):
        """
        Test that the gpu data is correct
        """
        self.assertEqual(self.rcm0.gpu_data(),
                         convert_gpu_metrics_to_data(self.gpu_metric_values))

    def test_non_gpu_data(self):
        """
        Test that the non-gpu data is correct
        """
        self.assertEqual(self.rcm0.non_gpu_data(), [
            convert_non_gpu_metrics_to_data(ngvm)
            for ngvm in self.rcm0_non_gpu_metric_values
        ])

    def test_data(self):
        """
        Test that the gpu + non-gpu data is correct
        """
        avg_gpu_data = convert_avg_gpu_metrics_to_data(
            self.avg_gpu_metric_values)

        data = [
            list(avg_gpu_data.values()) + convert_non_gpu_metrics_to_data(ngvm)
            for ngvm in self.rcm0_non_gpu_metric_values
        ]

        self.assertEqual(self.rcm0.data(), data)

    def test_gpus_used(self):
        """
        Test that the list of gpus used is correct
        """
        self.assertEqual(self.rcm0.gpus_used(), ['0', '1'])

    def test_get_gpu_metric(self):
        """
        Test that the gpu metric data is correct
        """
        avg_gpu_data = convert_avg_gpu_metrics_to_data(
            self.avg_gpu_metric_values)

        self.assertEqual(self.rcm0.get_gpu_metric("gpu_used_memory"),
                         avg_gpu_data["gpu_used_memory"])

        self.assertEqual(self.rcm0.get_gpu_metric("gpu_utilization"),
                         avg_gpu_data["gpu_utilization"])

    def test_get_non_gpu_metric(self):
        """
        Test that the non-gpu metric data is correct
        """
        non_gpu_data = [
            convert_non_gpu_metrics_to_data(non_gpu_metric_value)
            for non_gpu_metric_value in self.rcm0_non_gpu_metric_values
        ]

        self.assertEqual(self.rcm0.get_non_gpu_metric("perf_throughput"),
                         [non_gpu_data[0][0], non_gpu_data[1][0]])
        self.assertEqual(self.rcm0.get_non_gpu_metric("perf_latency_p99"),
                         [non_gpu_data[0][1], non_gpu_data[1][1]])
        self.assertEqual(self.rcm0.get_non_gpu_metric("cpu_used_ram"),
                         [non_gpu_data[0][2], non_gpu_data[1][2]])

    #TODO-TMA-569: Remove this test
    def test_get_metric_non_gpu(self):
        """
        Test that the non-gpu metric data is correct
        """
        non_gpu_data = [
            convert_non_gpu_metrics_to_data(non_gpu_metric_value)
            for non_gpu_metric_value in self.rcm0_non_gpu_metric_values
        ]

        self.assertEqual(self.rcm0.get_metric("perf_throughput"),
                         [non_gpu_data[0][0], non_gpu_data[1][0]])
        self.assertEqual(self.rcm0.get_metric("perf_latency_p99"),
                         [non_gpu_data[0][1], non_gpu_data[1][1]])
        self.assertEqual(self.rcm0.get_metric("cpu_used_ram"),
                         [non_gpu_data[0][2], non_gpu_data[1][2]])

    #TODO-TMA-569: Remove this test
    def test_get_metric_gpu(self):
        """
        Test that the gpu metric data is correct
        
        """
        avg_gpu_data = convert_avg_gpu_metrics_to_data(
            self.avg_gpu_metric_values)

        self.assertEqual(
            self.rcm0.get_metric("gpu_used_memory")[0],
            avg_gpu_data["gpu_used_memory"])

        self.assertEqual(
            self.rcm0.get_metric("gpu_utilization")[0],
            avg_gpu_data["gpu_utilization"])

    def test_get_weighted_non_gpu_metric(self):
        """
        Test that the weighted non-gpu metric data is correct
        """
        non_gpu_data = [
            convert_non_gpu_metrics_to_data(weighted_non_gpu_metric_value)
            for weighted_non_gpu_metric_value in
            self.rcm0_weighted_non_gpu_metric_values
        ]

        self.assertEqual(
            self.rcm0.get_weighted_non_gpu_metric("perf_throughput"),
            [non_gpu_data[0][0], non_gpu_data[1][0]])
        self.assertEqual(
            self.rcm0.get_weighted_non_gpu_metric("perf_latency_p99"),
            [non_gpu_data[0][1], non_gpu_data[1][1]])
        self.assertEqual(self.rcm0.get_weighted_non_gpu_metric("cpu_used_ram"),
                         [non_gpu_data[0][2], non_gpu_data[1][2]])

    def test_non_gpu_get_metric_value(self):
        """
        Test that the non-gpu metric value is correct
        """
        self.assertEqual(
            self.rcm0.get_non_gpu_metric_value("perf_throughput"),
            mean([
                self.rcm0_non_gpu_metric_values[0]['perf_throughput'],
                self.rcm0_non_gpu_metric_values[1]['perf_throughput']
            ]))

    def test_gpu_get_metric_value(self):
        """
        Test that the gpu metric value is correct
        """
        self.assertEqual(self.rcm0.get_gpu_metric_value("gpu_used_memory"),
                         self.avg_gpu_metric_values['gpu_used_memory'])

    #TODO-TMA-569: Remove this test
    def test_get_metric_value(self):
        """
        Test that the non-gpu metric value is correct
        """
        self.assertEqual(
            self.rcm0.get_metric_value("perf_throughput"),
            mean([
                self.rcm0_non_gpu_metric_values[0]['perf_throughput'],
                self.rcm0_non_gpu_metric_values[1]['perf_throughput']
            ]))

    def test_get_weighted_non_gpu_metric_value(self):
        """
        Test that the non-gpu weighted metric value is correct
        """
        weighted_metric_value = (
            (self.rcm0_non_gpu_metric_values[0]['perf_latency_p99'] *
             self.weights[0]) +
            (self.rcm0_non_gpu_metric_values[1]['perf_latency_p99'] *
             self.weights[1])) / sum(self.weights)

        self.assertEqual(
            self.rcm0.get_weighted_metric_value("perf_latency_p99"),
            weighted_metric_value)

    def test_model_specific_pa_params(self):
        """
        Test that the model specific PA params are correct
        """
        foo = self.rcm0.model_specific_pa_params()

        self.assertEqual(self.rcm0.model_specific_pa_params(),
                         self.model_specific_pa_params)

    def test_is_better_than(self):
        """
        Test to ensure measurement comparison is working as intended
        """
        # RCM0: 1000, 40    RCM1: 500, 30  weights:[1,3]
        # RCM0-A's throughput is better than RCM1-A (0.5)
        # RCM0-B's latency is worse than RCM1-B (-0.25)
        # Factoring in model config weighting
        # tips this is favor of RCM1 (0.125, -0.1875)
        self.assertFalse(self.rcm0.is_better_than(self.rcm1))

        # This tips the scale in the favor of RCM0 (0.2, -0.15)
        self.rcm0.set_model_config_weighting([2, 3])
        self.assertTrue(self.rcm0.is_better_than(self.rcm1))
        self.assertFalse(self.rcm0 < self.rcm1)

    def test_from_dict(self):
        """
        Test to ensure class can be correctly restored from a dictionary
        """
        rcm0_json = json.dumps(self.rcm0.__dict__, default=default_encode)

        rcm0_from_dict = RunConfigMeasurement.from_dict(json.loads(rcm0_json))

        self.assertEqual(rcm0_from_dict.model_variants_name(),
                         self.rcm0.model_variants_name())
        self.assertEqual(rcm0_from_dict.gpu_data(), self.rcm0.gpu_data())
        self.assertEqual(rcm0_from_dict.non_gpu_data(),
                         self.rcm0.non_gpu_data())
        self.assertEqual(rcm0_from_dict.data(), self.rcm0.data())
        self.assertEqual(rcm0_from_dict._model_config_measurements,
                         self.rcm0._model_config_measurements)

        # Catchall in case something new is added
        self.assertEqual(rcm0_from_dict, self.rcm0)

    def test_invert_values(self):
        """
        Test that GPU and non-GPU values are properly inverted
        """
        inverted_rcm0 = RunConfigMeasurement.invert_values(self.rcm0)

        inverted_rcm0_gpu_data = inverted_rcm0.gpu_data()
        rcm0_gpu_data = self.rcm0.gpu_data()

        for tag, value in rcm0_gpu_data.items():
            for index, gpu_data in enumerate(value):
                self.assertEqual(inverted_rcm0_gpu_data[tag][index]._value,
                                 -gpu_data._value)

    def _construct_rcm0(self):
        self.model_name = "modelA,modelB"
        self.model_config_name = ["modelA_config_0", "modelB_config_1"]
        self.model_variants_name = "".join(self.model_config_name)
        self.model_specific_pa_params = [{
            "batch_size": 1,
            "concurrency": 1
        }, {
            "batch_size": 2,
            "concurrency": 2
        }]

        self.gpu_metric_values = {
            '0': {
                "gpu_used_memory": 6000,
                "gpu_utilization": 60
            },
            '1': {
                "gpu_used_memory": 10000,
                "gpu_utilization": 20
            }
        }
        self.avg_gpu_metric_values = {
            "gpu_used_memory": 8000,
            "gpu_utilization": 40
        }

        self.rcm0_non_gpu_metric_values = [
            {
                # modelA_config_0
                "perf_throughput": 1000,
                "perf_latency_p99": 20,
                "cpu_used_ram": 1000
            },
            {
                # modelB_config_1
                "perf_throughput": 2000,
                "perf_latency_p99": 40,
                "cpu_used_ram": 1500
            }
        ]

        self.metric_objectives = [{
            "perf_throughput": 1
        }, {
            "perf_latency_p99": 1
        }]

        self.weights = [1, 3]

        self.rcm0_weighted_non_gpu_metric_values = []
        for index, non_gpu_metric_values in enumerate(
                self.rcm0_non_gpu_metric_values):
            self.rcm0_weighted_non_gpu_metric_values.append({
                objective: value * self.weights[index] / sum(self.weights)
                for (objective, value) in non_gpu_metric_values.items()
            })

        self.rcm0 = construct_run_config_measurement(
            self.model_name, self.model_config_name,
            self.model_specific_pa_params, self.gpu_metric_values,
            self.rcm0_non_gpu_metric_values, self.metric_objectives,
            self.weights)

    def _construct_rcm1(self):
        model_name = "modelA,modelB"
        model_config_name = ["modelA_config_2", "modelB_config_3"]
        model_variants_name = "".join(model_config_name)
        model_specific_pa_params = [{
            "batch_size": 3,
            "concurrency": 3
        }, {
            "batch_size": 4,
            "concurrency": 4
        }]

        gpu_metric_values = {
            '0': {
                "gpu_used_memory": 7000,
                "gpu_utilization": 40
            },
            '1': {
                "gpu_used_memory": 12000,
                "gpu_utilization": 30
            }
        }

        self.rcm1_non_gpu_metric_values = [
            {
                # modelA_config_2
                "perf_throughput": 500,
                "perf_latency_p99": 20,
                "cpu_used_ram": 1000
            },
            {
                # modelB_config_3
                "perf_throughput": 1200,
                "perf_latency_p99": 30,
                "cpu_used_ram": 1500
            }
        ]

        metric_objectives = [{"perf_throughput": 1}, {"perf_throughput": 1}]

        weights = [1, 3]

        self.rcm1_weighted_non_gpu_metric_values = []
        for index, non_gpu_metric_values in enumerate(
                self.rcm1_non_gpu_metric_values):
            self.rcm1_weighted_non_gpu_metric_values.append({
                objective: value * self.weights[index] / sum(weights)
                for (objective, value) in non_gpu_metric_values.items()
            })

        self.rcm1 = construct_run_config_measurement(
            model_name, model_config_name, model_specific_pa_params,
            gpu_metric_values, self.rcm1_non_gpu_metric_values,
            metric_objectives, weights)


if __name__ == '__main__':
    unittest.main()
