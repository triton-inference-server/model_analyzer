# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .common import test_result_collector as trc

from .mocks.mock_config import MockConfig
from .mocks.mock_model_config import MockModelConfig
from .mocks.mock_run_configs import MockRunConfigs

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.record.types.perf_throughput import PerfThroughput
from model_analyzer.model_manager import ModelManager
from model_analyzer.result.measurement import Measurement
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.triton.model.model_config import ModelConfig
from google.protobuf import json_format
from tritonclient.grpc import model_config_pb2

from unittest.mock import MagicMock
from unittest.mock import patch

from .common.test_utils import convert_to_bytes


class MetricsManagerSubclass(MetricsManager):
    """ 
    Overrides execute_run_configs() to gather a list of MockRunConfigs that
    contain the configured values of each would-be 'executed' run_config
    """

    def __init__(self, config, client, server, gpus, result_manager,
                 state_manager):
        super().__init__(config, client, server, gpus, result_manager,
                         state_manager)
        self._configs = MockRunConfigs()
        self._perf_throughput = 1

    def get_run_configs(self):
        """ Return the list of configs that would have been 'executed' """
        return self._configs

    def execute_run_config(self, config):
        self._configs.add_from_run_config(config)
        return self._get_next_measurements()

    def _get_next_measurements(self):
        """ Return fake measurements as if the run_configs had been executed """

        perf_throughput = PerfThroughput(self._get_next_perf_throughput_value())
        non_gpu_data = [perf_throughput]
        return Measurement(gpu_data=MagicMock(),
                           non_gpu_data=non_gpu_data,
                           perf_config=MagicMock())

    def _get_next_perf_throughput_value(self):
        self._perf_throughput *= 2
        return self._perf_throughput


class TestModelManager(trc.TestResultCollector):

    def __init__(self, methodname):
        super().__init__(methodname)
        self._args = [
            'model-analyzer', 'profile', '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file'
        ]

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

    def test_full_sweep(self):
        """
        Test a normal full sweep of options
        """
        expected_ranges = [{
            'instances': [1, 2, 3, 4, 5],
            'kind': ["KIND_GPU"],
            'batching': [0],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32, 64, 128]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32, 64, 128]
        }]

        yaml_content = convert_to_bytes("""
            profile_models: test_model
            run_config_search_max_concurrency: 128
            run_config_search_max_instance_count: 5
            run_config_search_disable: False
            """)

        self._test_model_manager(yaml_content, expected_ranges)

    def test_another_full_sweep(self):
        """
        Test another full sweep of options
        """

        expected_ranges = [{
            'instances': [1, 2, 3, 4, 5, 6, 7],
            'kind': ["KIND_GPU"],
            'batching': [0],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32]
        }]

        yaml_content = convert_to_bytes("""
            profile_models: test_model
            run_config_search_max_concurrency: 32
            run_config_search_max_instance_count: 7
            run_config_search_disable: False
            """)

        self._test_model_manager(yaml_content, expected_ranges)

    def test_run_search_disable(self):
        """
        Test with run_config_search_disable=True

        Expect 1 result that matches the default configuration because no manual 
        search options provided and automatic search disabled/ignored
        """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1]
        }]

        yaml_content = convert_to_bytes("""
            profile_models: test_model
            run_config_search_max_concurrency: 32
            run_config_search_max_instance_count: 7
            run_config_search_disable: True
            """)

        self._test_model_manager(yaml_content, expected_ranges)

    def test_manual_concurrency(self):
        """
        Test with manually specified concurrencies
        """
        expected_ranges = [{
            'instances': [1, 2, 3, 4, 5, 6, 7],
            'kind': ["KIND_GPU"],
            'batching': [0],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [5, 7]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [5, 7]
        }]

        yaml_content = convert_to_bytes("""
            profile_models: test_model
            run_config_search_max_concurrency: 32
            run_config_search_max_instance_count: 7
            run_config_search_disable: False
            concurrency: [5, 7]
            """)

        self._test_model_manager(yaml_content, expected_ranges)

    def test_remote_mode(self):
        """
        Test remote mode

        In remote mode all model_config_parameters (ie. instance count) are ignored
        """

        expected_ranges = [{
            'instances': [None],
            'kind': [None],
            'batching': [None],
            'batch_sizes': [1],
            'concurrency': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        }]

        yaml_content = convert_to_bytes("""
            profile_models: test_model
            run_config_search_max_concurrency: 512
            run_config_search_max_instance_count: 7
            run_config_search_disable: False
            triton_launch_mode: remote            
            """)

        self._test_model_manager(yaml_content, expected_ranges)

    def test_manual_parameters(self):
        """
        Test with manually specified concurrencies and batch sizes
        """

        expected_ranges = [{
            'instances': [1, 2, 3, 4, 5, 6, 7],
            'kind': ["KIND_GPU"],
            'batching': [0],
            'batch_sizes': [1, 2, 3],
            'max_batch_size': [8],
            'concurrency': [2, 10, 18, 26, 34, 42, 50, 58]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1, 2, 3],
            'max_batch_size': [8],
            'concurrency': [2, 10, 18, 26, 34, 42, 50, 58]
        }]

        yaml_content = convert_to_bytes("""
            profile_models: test_model
            run_config_search_max_concurrency: 512
            run_config_search_max_instance_count: 7
            run_config_search_disable: False
            concurrency:
                start: 2
                stop: 64
                step: 8
            batch_sizes: 1,2,3     
            """)

        self._test_model_manager(yaml_content, expected_ranges)

    def test_triton_parameters(self):
        """
        Test with manually specified triton options. 
        
        In this case we don't automatically search instances or dynamic_batching
        since model config parameters are specified.
        """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batch_sizes': [1],
            'max_batch_size': [1, 2, 4, 8, 16],
            'concurrency': [1, 2, 4, 8]
        }]

        yaml_content = convert_to_bytes("""
            run_config_search_max_concurrency: 8
            run_config_search_max_instance_count: 16
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,2,4,8,16]
            """)

        self._test_model_manager(yaml_content, expected_ranges)

    def test_default_config_always_run_no_dynamic_batching_off(self):
        """
        Test that the default config is run even when manual search excludes that case
        In this case, default config is (1 instance, max_batch_size=8, dynamic batching off)
        We should have a case of dynamic_batching off even though manual search only has it on
        """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [0],
            'max_queue_delay': ['200', '300'],
            'batch_sizes': [1],
            'max_batch_size': [1, 2, 4, 8, 16],
            'concurrency': [1, 2, 4, 8]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8]
        }]

        yaml_content = convert_to_bytes("""
            run_config_search_max_concurrency: 8
            run_config_search_max_instance_count: 16
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,2,4,8,16]
                        dynamic_batching:
                            max_queue_delay_microseconds: [200, 300]                        
            """)

        self._test_model_manager(yaml_content, expected_ranges)

    def test_default_config_always_run_wrong_instances(self):
        """
        Test that the default config is run even when manual search excludes that case
        In this case, default config is (2 instances, max_batch_size=8, dynamic batching off)
        We should have a 2-instance case even though manual search only has 1-instance
        """

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 2
            }
            ]
            """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_GPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }, {
            'instances': [2],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }]

        yaml_content = convert_to_bytes("""
            run_config_search_max_concurrency: 4
            run_config_search_max_instance_count: 16
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: 1
            """)

        self._test_model_manager(yaml_content, expected_ranges)

    def test_default_config_always_run_cpu_vs_gpu(self):
        """
        If the default configuration had KIND_CPU, make sure it is run (even if everything
        else is the same)
        """

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }, {
            'instances': [1],
            'kind': ["KIND_GPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }]

        yaml_content = convert_to_bytes("""
            run_config_search_max_concurrency: 4
            run_config_search_max_instance_count: 16
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: 1
            """)

        self._test_model_manager(yaml_content, expected_ranges)

    def test_default_config_always_run_automatic_search(self):
        """
        Test that the default config is run even when automatic search excludes that case
        In this case, default config is (4 instance, CPU, max_batch_size=8, dynamic batching off)
        We should have this 4 instance case though run_config_search_max_instance_count=1
        """

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 4
            }
            ]
            """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_GPU"],
            'batching': [0],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }, {
            'instances': [4],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }]

        yaml_content = convert_to_bytes("""
            run_config_search_max_concurrency: 4
            run_config_search_max_instance_count: 1
            run_config_search_disable: False
            profile_models: test_model
            """)
        self._test_model_manager(yaml_content, expected_ranges)

    def test_throughput_early_exit_minimum_runs(self):
        """
        Test that there is an early backoff when sweeping concurrency

        The behavior is that MA will try at least 4 concurrencies. If 
        at that point none of the last 3 attempts have had satisfactory 
        gain, it will stop

        This test hardcodes the 'throughput' to 1, so for all model
        configs the gain will be invalid and it will only try 4 
        concurrencies of (1,2,4,8) despite max_concurrency=128
        """

        expected_ranges = [{
            'instances': [1, 2],
            'kind': ["KIND_GPU"],
            'batching': [0],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8]
        }]

        yaml_content = convert_to_bytes("""
            profile_models: test_model
            run_config_search_max_concurrency: 128
            run_config_search_max_instance_count: 2
            run_config_search_disable: False
            """)

        with patch.object(MetricsManagerSubclass,
                          "_get_next_perf_throughput_value") as mock_method:
            mock_method.return_value = 1
            self._test_model_manager(yaml_content, expected_ranges)

    def test_throughput_early_exit(self):
        """
        Test that there is an early backoff when sweeping concurrency

        The behavior is that MA stop if it had 4 concurrencies in a row
        without any valid gain amongst any of them

        This test sets the 'throughput' to [1,2,4,8,16,16,16,16], which 
        will cause an early exit after trying the 8th concurrency (128)
        instead of searching all the way to 2048
        """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_GPU"],
            'batching': [0],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32, 64, 128]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32, 64, 128]
        }]

        yaml_content = convert_to_bytes("""
            profile_models: test_model
            run_config_search_max_concurrency: 2048
            run_config_search_max_instance_count: 1
            run_config_search_disable: False
            """)

        with patch.object(MetricsManagerSubclass,
                          "_get_next_perf_throughput_value") as mock_method:
            mock_method.side_effect = [
                1, 2, 4, 8, 16, 16, 16, 16, 1, 2, 4, 8, 16, 16, 16, 16
            ]
            self._test_model_manager(yaml_content, expected_ranges)

    def test_bad_result_early_exit(self):
        """
        Test that there is an early backoff for bad result (out of memory)

        If no measurements are returned in an attempt, no further concurrencies
        should be tried.

        This test hardcodes the measurements to be empty (bad result), so for all 
        model configs it will only try 1 concurrency despite max_concurrency=128
        """

        expected_ranges = [{
            'instances': [1, 2],
            'kind': ["KIND_GPU"],
            'batching': [0],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1]
        }]

        yaml_content = convert_to_bytes("""
            profile_models: test_model
            run_config_search_max_concurrency: 128
            run_config_search_max_instance_count: 2
            run_config_search_disable: False
            """)

        with patch.object(MetricsManagerSubclass,
                          "_get_next_measurements") as mock_method:
            mock_method.return_value = None
            self._test_model_manager(yaml_content, expected_ranges)

    def _test_model_manager(self, yaml_content, expected_ranges):
        """ 
        Test helper function that passes the given yaml_content into
        model_manager, runs the model, and confirms the result is as expected
        based on a full cartesian product of the lists in the input list of 
        dicts expected_ranges
        """

        # Use mock model config or else TritonModelAnalyzerException will be thrown as it tries to read from disk
        self.mock_model_config = MockModelConfig(self._model_config_protobuf)
        self.mock_model_config.start()
        config = self._evaluate_config(self._args, yaml_content)

        state_manager = AnalyzerStateManager(config, MagicMock())
        metrics_manager = MetricsManagerSubclass(config, MagicMock(),
                                                 MagicMock(), MagicMock(),
                                                 MagicMock(), state_manager)
        model_manager = ModelManager(config,
                                     MagicMock(), MagicMock(), metrics_manager,
                                     MagicMock(), state_manager)

        model_manager.run_models(config.profile_models)
        self.mock_model_config.stop()

        self._check_results(model_manager, expected_ranges)

    def _check_results(self, model_manager, expected_ranges):
        """ 
        Create a set of expected and actual run configs and confirm they are equal
        """
        run_configs = model_manager._metrics_manager.get_run_configs()
        expected_configs = MockRunConfigs()
        expected_configs.populate_from_ranges(expected_ranges)

        self.assertEqual(run_configs.get_configs_set(),
                         expected_configs.get_configs_set())

    def _evaluate_config(self, args, yaml_content):
        """ Parse the given yaml_content into a config and return it """

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

    @patch('model_analyzer.record.metrics_manager.MetricsManager.__init__',
           return_value=None)
    def test_is_config_in_results(self, mock_metrics_manager_init):
        """
        Tests that MetricsManager._is_config_in_results() works correctly.
        """

        metrics_manager = MetricsManager()

        model_i0_config = ModelConfig(
            json_format.ParseDict(
                {
                    'name': 'model_i0',
                    'instance_group': [{
                        'kind': 'KIND_GPU',
                        'count': 1
                    }]
                }, model_config_pb2.ModelConfig()))
        model_i1_config = ModelConfig(
            json_format.ParseDict(
                {
                    'name': 'model_i1',
                    'instance_group': [{
                        'kind': 'KIND_GPU',
                        'count': 2
                    }]
                }, model_config_pb2.ModelConfig()))
        model_results = {
            'model_i0': (model_i0_config,),
            'model_i1': (model_i1_config,)
        }

        model_config = json_format.ParseDict(
            {
                'name': 'model_i0',
                'instance_group': [{
                    'kind': 'KIND_GPU',
                    'count': 1
                }]
            }, model_config_pb2.ModelConfig())
        self.assertEqual(
            metrics_manager._is_config_in_results(model_config, model_results),
            True)

        model_config = json_format.ParseDict(
            {
                'name': 'model_i2',
                'instance_group': [{
                    'kind': 'KIND_GPU',
                    'count': 2
                }]
            }, model_config_pb2.ModelConfig())
        self.assertEqual(
            metrics_manager._is_config_in_results(model_config, model_results),
            False)

        model_config = json_format.ParseDict(
            {
                'name': 'model_i1',
                'max_batch_size': 1
            }, model_config_pb2.ModelConfig())
        self.assertEqual(
            metrics_manager._is_config_in_results(model_config, model_results),
            False)
