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

import unittest

from .common import test_result_collector as trc
from .common.test_utils import convert_to_bytes, ROOT_DIR
from .mocks.mock_config import MockConfig

from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_analyze \
    import ConfigCommandAnalyze
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager

import unittest
from unittest.mock import MagicMock, patch


class TestResultManager(trc.TestResultCollector):

    def setUp(self):
        args = [
            'model-analyzer', 'analyze', '-f', 'config.yml',
            '--checkpoint-directory', f'{ROOT_DIR}/multi-model-ckpt/'
        ]
        yaml_content = convert_to_bytes("""
            analysis_models: resnet50_libtorch,vgg19_libtorch
        """)
        config = self._evaluate_config(args, yaml_content)
        state_manager = AnalyzerStateManager(config=config, server=None)
        state_manager.load_checkpoint(checkpoint_required=True)

        self._multi_model_result_manager = ResultManager(
            config=config, state_manager=state_manager)

        self._num_measurements = 68
        self._multi_model_result_manager.create_tables()
        self._multi_model_result_manager.compile_and_sort_results()

    def tearDown(self):
        patch.stopall()

    def test_multi_model_row_count(self):
        """
        Check that a row was created for every measurement
        """
        self._multi_model_result_manager.tabulate_results()

        self.assertEqual(
            len(self._multi_model_result_manager.
                _result_tables['model_inference_metrics']._rows),
            self._num_measurements)

        self.assertEqual(
            len(self._multi_model_result_manager.
                _result_tables['model_gpu_metrics']._rows),
            self._num_measurements)

    def test_multi_model_gpu_metric_row_values(self):
        """
        Check that the values in a row are being outputted as expected
        """
        self._multi_model_result_manager.tabulate_results()

        # Values are extracted from checkpoint line 19012- 19460
        expected_row10_gpu_metric_values = [
            '"resnet50_libtorch,vgg19_libtorch"',  # Model
            'GPU-8557549f-9c89-4384-8bd6-1fd823c342e0',  # GPU UUID
            '"1,1"',  # Batch sizes
            '"1,2"',  # Concurrencies
            '"resnet50_libtorch_config_2,vgg19_libtorch_config_0"',  # Model config path
            '"2/GPU,1/GPU"',  # Instances
            "Yes",  # Satisfies constraints
            2527.0,  # GPU Memory
            70.6,  # GPU Utilization
            216.0  # GPU Power usage
        ]

        self.assertEqual(
            self._multi_model_result_manager.
            _result_tables['model_gpu_metrics']._rows[10],
            expected_row10_gpu_metric_values)

    def test_multi_model_inference_metric_row_values(self):
        """
        Check that the values in a row are being outputted as expected
        """
        self._multi_model_result_manager.tabulate_results()

        # Values are extracted from checkpoint lines 19012- 19460
        expected_row10_inference_metric_values = [
            '"resnet50_libtorch,vgg19_libtorch"',  # Model
            '"1,1"',  # Batch sizes
            '"1,2"',  # Concurrencies
            '"resnet50_libtorch_config_2,vgg19_libtorch_config_0"',  # Model config path
            '"2/GPU,1/GPU"',  # Instances
            "Yes",  # Satisfies constraints
            '"267.0,[105.0, 162.0]"',  # Throughput
            '"12.6,[11.4, 13.7]"'  # P99 Latency
        ]

        self.assertEqual(
            self._multi_model_result_manager.
            _result_tables['model_inference_metrics']._rows[10],
            expected_row10_inference_metric_values)

    def test_create_inference_table_with_backend_parameters(self):
        args = ['model-analyzer', 'analyze', '-f', 'config.yml']
        yaml_content = convert_to_bytes("""
            analysis_models: analysis_models
            inference_output_fields: model_name,batch_size,backend_parameter/parameter_1,backend_parameter/parameter_2
        """)
        config = self._evaluate_config(args, yaml_content)
        state_manager = AnalyzerStateManager(config=config, server=None)
        result_manager = ResultManager(config=config,
                                       state_manager=state_manager)

        result_manager._create_inference_table()
        self.assertTrue(result_manager._inference_output_fields == [
            'model_name', 'batch_size', 'backend_parameter/parameter_1',
            'backend_parameter/parameter_2'
        ])

    def test_get_common_row_items_with_backend_parameters(self):
        """
        This tests that a metrics model inference table row can be created with
        backend parameters included. Each backend parameter gets its own column.
        The column name is the backend parameter key (prepended with a prefix
        to avoid potentially overlapping with an existing column). The column
        value is the backend parameter value.

        Here is an example table:

        Models (Inference):
        Model     Model Config Path   backend_parameter/add_sub_key_1   backend_parameter/add_sub_key_2  
        add_sub   add_sub_config_2    add_sub_value_1                   add_sub_value_2                  
        add_sub   add_sub_config_0    add_sub_value_1                   add_sub_value_2                  
        add_sub   add_sub_config_1    add_sub_value_1                   add_sub_value_2                  

        Each row of the metrics model inference table corresponds to one model
        config variant.

        It is possible for a user to run the analyze command with multiple
        models config variants from different models with potentially different
        backend parameters. This test includes backend parameters from two
        separate models, showing that for one particular row (for a 'model A'
        config variant), it only populates the backend parameter cells for
        'model A', and the backend parameter cells for 'model B' are empty
        (None).

        Here is an example table with backend parameters from different models:

        Models (Inference):
        Model       Model Config Path   backend_parameter/add_sub_key_1   backend_parameter/add_sub_key_2   backend_parameter/add_sub_2_key_1   backend_parameter/add_sub_2_key_2  
        add_sub     add_sub_config_2    add_sub_value_1                   add_sub_value_2                   None                                None                               
        add_sub     add_sub_config_0    add_sub_value_1                   add_sub_value_2                   None                                None                               
        add_sub     add_sub_config_1    add_sub_value_1                   add_sub_value_2                   None                                None                               
        add_sub_2   add_sub_2_config_2  None                              None                              add_sub_2_value_1                   add_sub_2_value_2                  
        add_sub_2   add_sub_2_config_1  None                              None                              add_sub_2_value_1                   add_sub_2_value_2                  
        add_sub_2   add_sub_2_config_0  None                              None                              add_sub_2_value_1                   add_sub_2_value_2       
        """

        args = ['model-analyzer', 'analyze', '-f', 'config.yml']
        yaml_content = convert_to_bytes("""
            analysis_models: analysis_models
            inference_output_fields: model_name,batch_size,backend_parameter/model_1_key_1,backend_parameter/model_1_key_2,backend_parameter/model_2_key_1
        """)
        config = self._evaluate_config(args, yaml_content)
        state_manager = AnalyzerStateManager(config=config, server=None)
        result_manager = ResultManager(config=config,
                                       state_manager=state_manager)

        model_config_str = """
            parameters: {
            key: "model_1_key_1"
                value: {
                string_value:"model_1_value_1"
                }
            }
            parameters: {
            key:"model_1_key_2"
                value: {
                string_value:"model_1_value_2"
                }
            }
            """
        backend_parameters = text_format.Parse(
            model_config_str, model_config_pb2.ModelConfig()).parameters
        row = result_manager._get_common_row_items(
            fields=[
                'model_name', 'batch_size', 'backend_parameter/model_1_key_1',
                'backend_parameter/model_1_key_2',
                'backend_parameter/model_2_key_1'
            ],
            batch_sizes='batch_size',
            concurrencies=None,
            satisfies=None,
            model_name='model_name',
            model_config_path=None,
            dynamic_batchings=None,
            instance_groups=None,
            backend_parameters=backend_parameters)
        self.assertTrue(row == [
            'model_name', 'batch_size', 'model_1_value_1', 'model_1_value_2',
            None
        ])

    def _evaluate_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = ConfigCommandAnalyze()
        cli = CLI()
        cli.add_subcommand(
            cmd='analyze',
            help=
            'Collect and sort profiling results and generate data and summaries.',
            config=config)
        cli.parse()
        mock_config.stop()
        return config


if __name__ == "__main__":
    unittest.main()
