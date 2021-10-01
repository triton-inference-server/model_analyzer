# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.config.input.config_command_profile \
     import ConfigCommandProfile
from model_analyzer.config.run.run_search import RunSearch
from model_analyzer.cli.cli import CLI

from .mocks.mock_config import MockConfig
from .mocks.mock_os import MockOSMethods


class TestRunSearch(trc.TestResultCollector):

    def _evaluate_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = ConfigCommandProfile()
        cli = CLI()
        cli.add_subcommand(
            cmd='profile',
            help=
            'Run model inference profiling based on specified CLI or config options.',
            config=config)
        cli.parse()
        mock_config.stop()
        return config

    def _create_measurement(self, value):
        metric_attrs = {'value': MagicMock(return_value=value)}
        measurement_attrs = {
            'get_metric': MagicMock(return_value=MagicMock(**metric_attrs)),
            'get_metric_value': MagicMock(return_value=value)
        }
        return MagicMock(**measurement_attrs)

    def setUp(self):
        # Mock path validation
        self.mock_os = MockOSMethods(
            mock_paths=['model_analyzer.config.input.config_utils'])
        self.mock_os.start()

    def tearDown(self):
        self.mock_os.stop()

    def test_run_search(self):
        args = [
            'model-analyzer', 'profile', '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file', '--profile-models', 'vgg11'
        ]

        yaml_content = """
            run_config_search_max_concurrency: 128
            run_config_search_max_preferred_batch_size: 16
            run_config_search_max_instance_count: 5
            concurrency: []
            profile_models:
                - my-model
            """

        config = self._evaluate_config(args, yaml_content)
        run_search = RunSearch(config=config)

        concurrencies = config.profile_models[0].parameters()['concurrency']
        run_search.init_model_sweep(concurrencies, True)
        config_model, model_sweeps = run_search.get_model_sweep(
            config.profile_models[0])

        start_throughput = 2
        expected_concurrency = 1
        expected_instance_count = 1
        while model_sweeps:
            model_sweep = model_sweeps.pop()
            current_concurrency = config_model.parameters()['concurrency'][0]
            self.assertEqual(expected_concurrency, current_concurrency)
            run_search.add_measurements(
                [self._create_measurement(start_throughput)])
            start_throughput *= 2
            expected_concurrency *= 2
            current_instance_count = model_sweep['instance_group'][0]['count']

            self.assertEqual(current_instance_count, expected_instance_count)
            if expected_concurrency > config.run_config_search_max_concurrency:
                expected_concurrency = 1
                expected_instance_count += 1
                if expected_instance_count > config.run_config_search_max_instance_count:
                    expected_instance_count = 1

            config_model, model_sweeps = run_search.get_model_sweep(
                config_model)

    def test_run_search_failing(self):
        args = [
            'model-analyzer', 'profile', '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file', '--profile-models', 'vgg11'
        ]

        yaml_content = """
            run_config_search_max_concurrency: 128
            run_config_search_max_preferred_batch_size: 16
            run_config_search_max_instance_count: 5
            concurrency: []
            profile_models:
                - my-model
            """

        config = self._evaluate_config(args, yaml_content)
        run_search = RunSearch(config=config)

        # Simulate running multiple sweeps
        for i in range(2):
            concurrencies = config.profile_models[0].parameters()['concurrency']
            run_search.init_model_sweep(concurrencies, True)
            config_model, model_sweeps = run_search.get_model_sweep(
                config.profile_models[0])

            start_throughput = 2
            expected_concurrency = 1
            expected_instance_count = 1
            total_runs = 0
            while model_sweeps:
                model_sweep = model_sweeps.pop()
                current_concurrency = config_model.parameters(
                )['concurrency'][0]
                run_search.add_measurements(
                    [self._create_measurement(start_throughput)])
                start_throughput *= 1.02
                self.assertEqual(expected_concurrency, current_concurrency)
                current_instance_count = model_sweep['instance_group'][0][
                    'count']

                self.assertEqual(current_instance_count,
                                 expected_instance_count)
                total_runs += 1

                # Because the growth of throughput is not substantial, the algorithm
                # will stop execution.
                if total_runs == 4:
                    total_runs = 0
                    expected_concurrency = 1
                    expected_instance_count += 1
                    if expected_instance_count > config.run_config_search_max_instance_count:
                        expected_instance_count = 1
                else:
                    expected_concurrency *= 2

                config_model, model_sweeps = run_search.get_model_sweep(
                    config_model)
