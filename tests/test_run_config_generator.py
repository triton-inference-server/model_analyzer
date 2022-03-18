# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.config.generate.run_config_generator import RunConfigGenerator
from model_analyzer.config.input.config_command_profile \
     import ConfigCommandProfile
from model_analyzer.cli.cli import CLI
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.record.types.perf_throughput import PerfThroughput
from model_analyzer.result.measurement import Measurement
from .common import test_result_collector as trc
from .common.test_utils import convert_to_bytes
from .mocks.mock_config import MockConfig
from .mocks.mock_model_config import MockModelConfig
from .mocks.mock_os import MockOSMethods
from unittest.mock import MagicMock

from model_analyzer.config.generate.generator_utils import GeneratorUtils as utils

from model_analyzer.config.input.config_defaults import \
    DEFAULT_BATCH_SIZES, DEFAULT_TRITON_LAUNCH_MODE, \
    DEFAULT_CLIENT_PROTOCOL, DEFAULT_TRITON_INSTALL_PATH, DEFAULT_OUTPUT_MODEL_REPOSITORY, \
    DEFAULT_TRITON_INSTALL_PATH, DEFAULT_OUTPUT_MODEL_REPOSITORY, \
    DEFAULT_TRITON_HTTP_ENDPOINT, DEFAULT_TRITON_GRPC_ENDPOINT, DEFAULT_MEASUREMENT_MODE, \
    DEFAULT_RUN_CONFIG_MAX_CONCURRENCY, DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT, DEFAULT_RUN_CONFIG_MAX_MODEL_BATCH_SIZE


class TestRunConfigGenerator(trc.TestResultCollector):

    def __init__(self, methodname):
        super().__init__(methodname)
        self._fake_throughput = 1

    def test_default_config_single_model(self):
        """
        Test Default Single Model:  
        
        num_PAC = log2(DEFAULT_RUN_CONFIG_MAX_CONCURRENCY) + 1
        num_MC = (  DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT 
                  x log2(DEFAULT_RUN_CONFIG_MAX_MODEL_BATCH_SIZE)
                 ) + 1
        total = (num_PAC * num_MC) will be generated by the auto-search
        """

        # yapf: disable
        yaml_content = convert_to_bytes("""
            profile_models:
                - my-model
            """)
        # yapf: enable

        expected_pa_configs = len(
            utils.generate_log2_list(DEFAULT_RUN_CONFIG_MAX_CONCURRENCY))

        expected_model_configs = DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT \
                               * len(utils.generate_log2_list(DEFAULT_RUN_CONFIG_MAX_MODEL_BATCH_SIZE)) \
                               + 1
        expected_num_of_configs = expected_pa_configs * expected_model_configs

        self._run_and_test_run_config_generator(
            yaml_content, expected_config_count=expected_num_of_configs)

    def test_two_models(self):
        """
        Test Two Models that have the same automatic configuration:
        
        num_PAC = 2
        num_MC = (2 * 2) + 1 = 5
        model_total = (2 * 5) = 10 will be generated by the auto-search for each model

        total = model_total * model_total = 100
        """

        # yapf: disable
        yaml_content = convert_to_bytes("""
            run_config_search_max_model_batch_size: 2
            run_config_search_max_instance_count: 2
            run_config_search_max_concurrency: 2
            profile_models:
                - my-model
                - my-model2

            """)
        # yapf: enable

        expected_num_of_configs = 100

        self._run_and_test_run_config_generator(
            yaml_content, expected_config_count=expected_num_of_configs)

    def test_two_uneven_models(self):
        """
        Test Two Uneven Models:
        
        Model 1 is auto search:
            num_PAC = 3
            num_MC = (2 * 2) + 1 = 5
            model1_total = (3 * 5) = 15

        Model 2 is manual search:
            num_PAC = 2
            num_MC = (2 * 3) + 1 = 7
            model2_total = (2 * 7) = 14
        total = model1_total * model2_total = 210
        """

        # yapf: disable
        yaml_content = convert_to_bytes("""
            run_config_search_max_model_batch_size: 2
            run_config_search_max_instance_count: 2
            run_config_search_max_concurrency: 2
            profile_models:
                my-model:
                    parameters:
                        concurrency: [1,2,3]  
                my-model2:
                    model_config_parameters:
                        max_batch_size: [1,4,16]
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1,2]
            """)
        # yapf: enable

        expected_num_of_configs = 210

        self._run_and_test_run_config_generator(
            yaml_content, expected_config_count=expected_num_of_configs)

    def test_three_uneven_models(self):
        """
        Test Three Uneven Models:
        
        Model 1 is auto search:
            num_PAC = 2
            num_MC = (2 * 2) + 1 = 5
            model1_total = (2 * 5) = 10

        Model 2 is auto search with fixed concurrency:
            num_PAC = 5
            num_MC = (2 * 2) + 1 = 5
            model1_total = (5 * 5) = 25

        Model 3 is manual search:
            num_PAC = 3
            num_MC = (3 * 4) + 1 = 13
            model2_total = (3 * 13) = 39
        total = model1_total * model2_total * model3_total = 9750
        """

        # yapf: disable
        yaml_content = convert_to_bytes("""
            run_config_search_max_model_batch_size: 2
            run_config_search_max_instance_count: 2
            run_config_search_max_concurrency: 2
            profile_models:
                - my-model
                - 
                  my-model2:
                    parameters:
                        concurrency: [1,3,5,7,9]  
                -
                  my-model3:
                    parameters:
                        concurrency: [10, 20, 30]
                    model_config_parameters:
                        max_batch_size: [1,4,16,64]
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1,2,3]
            """)
        # yapf: enable

        expected_num_of_configs = 9750

        self._run_and_test_run_config_generator(
            yaml_content, expected_config_count=expected_num_of_configs)

    def _run_and_test_run_config_generator(self, yaml_content,
                                           expected_config_count):
        args = [
            'model-analyzer', 'profile', '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file'
        ]

        protobuf = """
            max_batch_size: 8
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

        self.mock_model_config = MockModelConfig(protobuf)
        self.mock_model_config.start()
        config = self._evaluate_config(args, yaml_content)

        rcg = RunConfigGenerator(config, config.profile_models, MagicMock())

        run_configs = []
        rcg_config_generator = rcg.next_config()
        while not rcg.is_done():
            run_configs.append(next(rcg_config_generator))
            rcg.set_last_results(self._get_next_fake_results())

        self.assertEqual(expected_config_count, len(set(run_configs)))

        # Checks that each ModelRunConfig contains the expected number of model_configs
        for run_config in run_configs:
            self.assertEqual(len(run_config.model_run_configs()),
                             len(config.profile_models))

        self.mock_model_config.stop()

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

    def setUp(self):
        # Mock path validation
        self.mock_os = MockOSMethods(
            mock_paths=['model_analyzer.config.input.config_utils'])
        self.mock_os.start()

    def tearDown(self):
        self.mock_os.stop()

    def _get_next_fake_results(self):
        self._fake_throughput *= 2
        perf_throughput = PerfThroughput(self._fake_throughput)
        measurement = Measurement(gpu_data=MagicMock(),
                                  non_gpu_data=[perf_throughput],
                                  perf_config=MagicMock())
        return [measurement]
