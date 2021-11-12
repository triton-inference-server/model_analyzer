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


from .common import test_result_collector as trc

from .mocks.mock_config import MockConfig
from .mocks.mock_os import MockOSMethods
from .mocks.mock_model_config import MockModelConfig
from .mocks.mock_client import MockTritonClientMethods

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.run.run_search import RunSearch
from model_analyzer.config.run.run_config_generator import RunConfigGenerator
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.model_manager import ModelManager
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager

import itertools
import logging
import sys

from unittest.mock import MagicMock
from unittest.mock import patch



class ModelManagerSubclass(ModelManager):
    """ 
    Overrides execute_run_configs() to gather a list of tuples that contain 
    the main values of each 'executed' run_config
    """

    def __init__(self, config, client, server, metrics_manager, result_manager, state_manager):
        super().__init__(config, client, server, metrics_manager, result_manager, state_manager)
        self._configs = []
    
    def _execute_run_configs(self):
        while self._run_config_generator.run_configs():
            config = self._run_config_generator.next_config()
            model_config = config.model_config().to_dict()
            perf_config = config.perf_config()

            instances = None
            if model_config.get("instanceGroup") is not None:
                instances = model_config["instanceGroup"][0]["count"]

            dynamic_batching = None
            if model_config.get("dynamicBatching") is not None:
                dynamic_batching = model_config["dynamicBatching"].get("preferredBatchSize", [0])[0]

            batch_size = perf_config.__getitem__("batch-size")
            concurrency = perf_config.__getitem__("concurrency-range")

            self._configs.append((instances, dynamic_batching, batch_size, concurrency))

    def get_run_configs(self):
        return self._configs


@patch('model_analyzer.config.run.run_search.RunSearch.add_measurements', MagicMock())
class TestModelManager(trc.TestResultCollector):

    def test_full_sweep(self):
        """
        Test a normal full sweep of options
        """

        expected_instances = [1,2,3,4,5]
        expected_batching = [None,0,1,2,4,8,16]
        expected_batch_sizes = [1]
        expected_concurrency = [1,2,4,8,16,32,64,128]

        yaml_content = """
            run_config_search_max_concurrency: 128
            run_config_search_max_preferred_batch_size: 16
            run_config_search_max_instance_count: 5
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            """   

        expected_values = [expected_instances, expected_batching, expected_batch_sizes, expected_concurrency]
        self._test_model_manager(yaml_content, expected_values)


    def test_another_full_sweep(self):
        """
        Test another full sweep of options
        """

        expected_instances = [1,2,3,4,5,6,7]
        expected_batching = [None,0,1,2,4,8]
        expected_batch_sizes = [1]
        expected_concurrency = [1,2,4,8,16,32]

        yaml_content = """
            run_config_search_max_concurrency: 32
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            """   

        expected_values = [expected_instances, expected_batching, expected_batch_sizes, expected_concurrency]
        self._test_model_manager(yaml_content, expected_values)


    def test_preferred_batch_size_disable(self):
        """
        Test with search_preferred_batch_size_disable=True
        """

        expected_instances = [1,2,3,4,5,6,7]
        expected_batching = [None]
        expected_batch_sizes = [1]
        expected_concurrency = [1,2,4,8,16,32]

        yaml_content = """
            run_config_search_max_concurrency: 32
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : True
            run_config_search_disable: False
            """   

        expected_values = [expected_instances, expected_batching, expected_batch_sizes, expected_concurrency]
        self._test_model_manager(yaml_content, expected_values)


    def test_run_search_disable(self):
        """
        Test with run_config_search_disable=True

        Expect 1 result because no manual search options provided and automatic search disabled/ignored
        """

        expected_instances = [None]
        expected_batching = [None]
        expected_batch_sizes = [1]
        expected_concurrency = [1]

        yaml_content = """
            run_config_search_max_concurrency: 32
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: True
            """   

        expected_values = [expected_instances, expected_batching, expected_batch_sizes, expected_concurrency]
        self._test_model_manager(yaml_content, expected_values)


    def test_manual_concurrency(self):
        """
        Test with manually specified concurrencies
        """

        expected_instances = [1,2,3,4,5,6,7]
        expected_batching = [None,0,1,2,4,8]
        expected_batch_sizes = [1]
        expected_concurrency = [5,7]


        yaml_content = """
            run_config_search_max_concurrency: 32
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            concurrency: [5, 7]
            """   

        expected_values = [expected_instances, expected_batching, expected_batch_sizes, expected_concurrency]
        self._test_model_manager(yaml_content, expected_values)


    def test_remote_mode(self):
        """
        Test remote mode

        In remote mode all model_config_parameters (preferred_batch_size, instance count) are ignored
        """

        expected_instances = [None]
        expected_batching = [None]
        expected_batch_sizes = [1]
        expected_concurrency = [1,2,4,8,16,32,64,128,256,512]

        yaml_content = """
            run_config_search_max_concurrency: 512
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            triton_launch_mode: remote            
            """   

        expected_values = [expected_instances, expected_batching, expected_batch_sizes, expected_concurrency]
        self._test_model_manager(yaml_content, expected_values)



    def test_manual_parameters(self):
        """
        Test with manually specified concurrencies and batch sizes
        """

        expected_instances = [1,2,3,4,5,6,7]
        expected_batching = [None, 0, 1, 2, 4, 8]
        expected_batch_sizes = [1,2,3]
        expected_concurrency = [2,10,18,26,34,42,50,58]

        yaml_content = """
            run_config_search_max_concurrency: 512
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            concurrency:
                start: 2
                stop: 64
                step: 8
            batch_sizes: 1,2,3          
            """   

        expected_values = [expected_instances, expected_batching, expected_batch_sizes, expected_concurrency]
        self._test_model_manager(yaml_content, expected_values)


    def _test_model_manager(self, yaml_content, expected_values):
        """ 
        Test helper function that passes the given yaml_content into
        model_manager and asserts that the number of run_configs generated
        is equal to the passed in expected_count 
        """

        args = [
            'model-analyzer', 'profile', '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file', '--profile-models', 'vgg11'
        ]

        config = self._evaluate_config(args, yaml_content)
        state_manager = AnalyzerStateManager(config, MagicMock())
        model_manager = ModelManagerSubclass(config, MagicMock(), MagicMock(), MagicMock(), MagicMock(), state_manager)

        model_manager.run_model(config.profile_models[0])

        run_configs = model_manager.get_run_configs()
        run_configs_set = set(run_configs)
        
        expected_set = set(itertools.product(*expected_values))

        # Confirm that the full set of configs is exactly the same as expected
        #
        self.assertEqual(expected_set, run_configs_set)


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

    def setUp(self):
        # Use mock model config or else TritonModelAnalyzerException will be thrown as it tries to read from disc
        self.mock_model_config = MockModelConfig()
        self.mock_model_config.start()

    def tearDown(self):
        self.mock_model_config.stop()

