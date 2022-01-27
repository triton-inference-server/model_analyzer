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

from .common import test_result_collector as trc
from .common.test_utils import convert_to_bytes
from .mocks.mock_config import MockConfig

from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_analyze \
    import ConfigCommandAnalyze
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager


class TestResultManager(trc.TestResultCollector):

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
            batch_size='batch_size',
            concurrency=None,
            satisfies=None,
            model_name='model_name',
            model_config_path=None,
            dynamic_batching=None,
            instance_group=None,
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
