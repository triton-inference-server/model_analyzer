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

from unittest.mock import MagicMock
from .common import test_result_collector as trc

from model_analyzer.config.run.run_search import RunSearch
from model_analyzer.config.input.objects.config_model import ConfigModel


class TestRunSearch(trc.TestResultCollector):
    def _create_throughput(self, value):
        throughput = MagicMock()
        throughput.get_value_of_metric().value.return_value = value
        return throughput

    def test_run_search(self):
        max_concurrency = 128
        max_preferred_batch_size = 16
        max_instance_count = 5
        run_search = RunSearch(max_concurrency, max_instance_count,
                               max_preferred_batch_size)

        config_model = ConfigModel('my-model', parameters={'concurrency': []})
        run_search.init_model_sweep(config_model, True)
        config_model, model_sweeps = run_search.get_model_sweeps(
            config_model)

        start_throughput = 2
        expected_concurrency = 1
        expected_instance_count = 1
        while model_sweeps:
            model_sweep = model_sweeps.pop()
            current_concurrency = config_model.parameters()['concurrency'][0]
            self.assertEqual(expected_concurrency, current_concurrency)
            run_search.add_run_results(
                {'*': [self._create_throughput(start_throughput)]})
            start_throughput *= 2
            expected_concurrency *= 2
            current_instance_count = model_sweep['instance_group'][0]['count']

            self.assertEqual(current_instance_count, expected_instance_count)
            if expected_concurrency > max_concurrency:
                expected_concurrency = 1
                expected_instance_count += 1
                if expected_instance_count > max_instance_count:
                    expected_instance_count = 1

            config_model, model_sweeps = run_search.get_model_sweeps(
                config_model)

    def test_run_search_failing(self):
        max_concurrency = 128
        max_preferred_batch_size = 16
        max_instance_count = 5
        run_search = RunSearch(max_concurrency, max_instance_count,
                               max_preferred_batch_size)

        config_model = ConfigModel('my-model', parameters={'concurrency': []})
        run_search.init_model_sweep(config_model, True)
        config_model, model_sweeps = run_search.get_model_sweeps(
            config_model)

        start_throughput = 2
        expected_concurrency = 1
        expected_instance_count = 1
        total_runs = 0
        while model_sweeps:
            model_sweep = model_sweeps.pop()
            current_concurrency = config_model.parameters()['concurrency'][0]
            run_search.add_run_results(
                {'*': [self._create_throughput(start_throughput)]})
            start_throughput *= 1.02
            self.assertEqual(expected_concurrency, current_concurrency)
            current_instance_count = model_sweep['instance_group'][0]['count']

            self.assertEqual(current_instance_count, expected_instance_count)
            total_runs += 1

            # Because the growth of throughput is not substantial, the algorithm
            # will stop execution.
            if total_runs == 4:
                total_runs = 0
                expected_concurrency = 1
                expected_instance_count += 1
                if expected_instance_count > max_instance_count:
                    expected_instance_count = 1
            else:
                expected_concurrency *= 2

            config_model, model_sweeps = run_search.get_model_sweeps(
                config_model)
