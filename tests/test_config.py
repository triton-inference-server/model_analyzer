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

import unittest
from .mocks.mock_config import MockConfig
from .common import test_result_collector as trc
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from model_analyzer.cli.cli import CLI
from model_analyzer.config.config import AnalyzerConfig
from model_analyzer.config.config_model import ConfigModel
from model_analyzer.config.config_list_string import ConfigListString
from model_analyzer.config.config_list_generic import ConfigListGeneric
from model_analyzer.config.config_primitive import ConfigPrimitive
from model_analyzer.config.config_union import ConfigUnion
from model_analyzer.config.config_object import ConfigObject
from model_analyzer.config.config_enum import ConfigEnum
from model_analyzer.config.config_list_numeric import \
    ConfigListNumeric


class TestConfig(trc.TestResultCollector):
    def _evaluate_config(self, args, yaml_content):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)
        cli.parse()
        mock_config.stop()

        return config

    def _assert_equality_of_model_configs(self, model_configs,
                                          expected_model_configs):
        self.assertTrue(len(model_configs) == len(expected_model_configs))
        for model_config, expected_model_config \
                in zip(model_configs, expected_model_configs):
            self.assertTrue(expected_model_config.model_name() ==
                            model_config.model_name())
            self.assertTrue(expected_model_config.parameters() ==
                            model_config.parameters())
            self.assertTrue(expected_model_config.constraints() ==
                            model_config.constraints())
            self.assertTrue(expected_model_config.objectives() ==
                            model_config.objectives())
            self.assertTrue(expected_model_config.model_config_parameters() ==
                            model_config.model_config_parameters())

    def _assert_model_config_types(self, model_config):
        self.assertIsInstance(model_config.field_type(), ConfigUnion)

        if isinstance(model_config.field_type().raw_value(),
                      ConfigListGeneric):
            self.assertIsInstance(
                model_config.field_type().raw_value().container_type(),
                ConfigUnion)
        else:
            self.assertIsInstance(model_config.field_type().raw_value(),
                                  ConfigObject)

    def _assert_model_object_types(self,
                                   model_config,
                                   model_name,
                                   check_parameters=False,
                                   check_concurrency=False,
                                   check_batch_size=False):

        if isinstance(model_config, ConfigUnion):
            self.assertIsInstance(model_config.raw_value(), ConfigObject)
            self.assertIsInstance(
                model_config.raw_value().raw_value()[model_name], ConfigObject)
            model_config = model_config.raw_value().raw_value()[model_name]
        else:
            self.assertIsInstance(model_config, ConfigObject)
            if check_parameters:
                parameters_config = model_config.raw_value()['parameters']
                self.assertIsInstance(parameters_config, ConfigObject)

                if check_concurrency:
                    self.assertIsInstance(
                        parameters_config.raw_value()['concurrency'],
                        ConfigListNumeric)

    def _assert_model_config_params(self, model_config_parameters):
        self.assertIsInstance(model_config_parameters, ConfigObject)

        input_param = model_config_parameters.raw_value()['input']
        self.assertIsInstance(input_param, ConfigUnion)
        self.assertIsInstance(input_param.raw_value(), ConfigListGeneric)
        self.assertIsInstance(input_param.raw_value().container_type(),
                              ConfigObject)

        # Check types for 'name'
        name_param = input_param.raw_value().raw_value()[0].raw_value()['name']
        self.assertIsInstance(name_param, ConfigUnion)
        self.assertIsInstance(name_param.raw_value(), ConfigPrimitive)

        # Check types for 'data_type'
        data_type_param = input_param.raw_value().raw_value()[0].raw_value(
        )['data_type']
        self.assertIsInstance(data_type_param, ConfigUnion)
        self.assertIsInstance(data_type_param.raw_value(), ConfigEnum)

        # Check types for 'dims'
        dims_param = input_param.raw_value().raw_value()[0].raw_value()['dims']
        self.assertIsInstance(dims_param, ConfigUnion)
        self.assertIsInstance(dims_param.raw_value(), ConfigListGeneric)
        self.assertIsInstance(dims_param.raw_value().container_type(),
                              ConfigPrimitive)

        # Check types for 'format'
        format_param = input_param.raw_value().raw_value()[0].raw_value(
        )['format']
        self.assertIsInstance(format_param, ConfigUnion)
        self.assertIsInstance(format_param.raw_value(), ConfigEnum)

    def _assert_model_str_type(self, model_config):
        self.assertIsInstance(model_config, ConfigUnion)
        self.assertIsInstance(model_config.raw_value(), ConfigPrimitive)

    def test_config(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file', '--model-names', 'vgg11'
        ]
        yaml_content = 'model_repository: yaml_repository'
        config = self._evaluate_config(args, yaml_content)

        # CLI flag has the highest priority
        self.assertTrue(
            config.get_all_config()['model_repository'] == 'cli_repository')

        args = [
            'model-analyzer', '-f', 'path-to-config-file', '--model-names',
            'vgg11'
        ]
        yaml_content = 'model_repository: yaml_repository'
        config = self._evaluate_config(args, yaml_content)

        # If CLI flag doesn't exist, YAML config has the highest priority
        self.assertTrue(
            config.get_all_config()['model_repository'] == 'yaml_repository')

        args = ['model-analyzer', '-f', 'path-to-config-file']
        yaml_content = 'model_repository: yaml_repository'
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)

        # When a required field is not specified, parse will lead to an
        # exception
        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()

        mock_config.stop()

    def test_range_and_list_values(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]
        yaml_content = 'model_names: model_1,model_2'
        config = self._evaluate_config(args, yaml_content)
        expected_model_configs = [
            ConfigModel('model_1'),
            ConfigModel('model_2')
        ]
        self._assert_equality_of_model_configs(
            config.get_all_config()['model_names'], expected_model_configs)
        self.assertIsInstance(
            config.get_config()['model_names'].field_type().raw_value(),
            ConfigListString)

        yaml_content = """
model_names:
    - model_1
    - model_2
"""
        config = self._evaluate_config(args, yaml_content)
        self._assert_equality_of_model_configs(
            config.get_all_config()['model_names'], expected_model_configs)

        model_config = config.get_config()['model_names']
        self._assert_model_config_types(model_config)
        self.assertIsInstance(
            model_config.field_type().raw_value().raw_value()[0].raw_value(),
            ConfigPrimitive)

        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file', '--model-names', 'model_1,model_2'
        ]
        yaml_content = """
batch_sizes:
    - 2
    - 3
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertTrue(config.get_all_config()['batch_sizes'] == [2, 3])
        self.assertIsInstance(config.get_config()['batch_sizes'].field_type(),
                              ConfigListNumeric)

        yaml_content = """
batch_sizes: 2
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertTrue(config.get_all_config()['batch_sizes'] == [2])
        self.assertIsInstance(config.get_config()['batch_sizes'].field_type(),
                              ConfigListNumeric)

        yaml_content = """
concurrency: 2
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertTrue(config.get_all_config()['concurrency'] == [2])
        self.assertIsInstance(config.get_config()['concurrency'].field_type(),
                              ConfigListNumeric)
        self.assertTrue(config.get_all_config()['batch_sizes'] == [1])
        self.assertIsInstance(config.get_config()['batch_sizes'].field_type(),
                              ConfigListNumeric)

        yaml_content = """
batch_sizes:
    start: 2
    stop: 6
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertTrue(
            config.get_all_config()['batch_sizes'] == [2, 3, 4, 5, 6])
        self.assertIsInstance(config.get_config()['batch_sizes'].field_type(),
                              ConfigListNumeric)

        yaml_content = """
batch_sizes:
    start: 2
    stop: 6
    step: 2
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertTrue(config.get_all_config()['batch_sizes'] == [2, 4, 6])
        self.assertIsInstance(config.get_config()['batch_sizes'].field_type(),
                              ConfigListNumeric)

    def test_object(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
          - 1
          - 2
          - 3
          - 4
  - vgg_19_graphdef
"""
        config = self._evaluate_config(args, yaml_content)
        model_config = config.get_config()['model_names']
        self._assert_model_config_types(model_config)

        expected_model_objects = [
            ConfigModel('vgg_16_graphdef',
                        parameters={'concurrency': [1, 2, 3, 4]}),
            ConfigModel('vgg_19_graphdef')
        ]

        # Check the types for the first value
        first_model = model_config.field_type().raw_value().raw_value()[0]
        self._assert_model_object_types(first_model,
                                        'vgg_16_graphdef',
                                        check_parameters=True,
                                        check_concurrency=True)

        # Check the types for the second value
        second_model = model_config.field_type().raw_value().raw_value()[1]
        self._assert_model_str_type(second_model)

        self._assert_equality_of_model_configs(
            config.get_all_config()['model_names'], expected_model_objects)

        yaml_content = """
model_names:
  vgg_16_graphdef:
    parameters:
      concurrency:
        - 1
        - 2
        - 3
        - 4
  vgg_19_graphdef:
    parameters:
      concurrency:
        - 1
        - 2
        - 3
        - 4
      batch_sizes:
          start: 2
          stop: 6
          step: 2
"""
        expected_model_objects = [
            ConfigModel('vgg_16_graphdef',
                        parameters={'concurrency': [1, 2, 3, 4]}),
            ConfigModel('vgg_19_graphdef',
                        parameters={
                            'concurrency': [1, 2, 3, 4],
                            'batch_sizes': [2, 4, 6]
                        })
        ]
        config = self._evaluate_config(args, yaml_content)
        self._assert_equality_of_model_configs(
            config.get_all_config()['model_names'], expected_model_objects)

        model_config = config.get_config()['model_names']
        self._assert_model_config_types(model_config)

        # first model
        first_model = model_config.field_type().raw_value().raw_value(
        )['vgg_16_graphdef']
        self._assert_model_object_types(first_model,
                                        'vgg_16_graphdef',
                                        check_parameters=True,
                                        check_concurrency=True,
                                        check_batch_size=True)

        # second model
        second_model = model_config.field_type().raw_value().raw_value(
        )['vgg_19_graphdef']
        self._assert_model_object_types(second_model,
                                        'vgg_19_graphdef',
                                        check_parameters=True,
                                        check_concurrency=True,
                                        check_batch_size=True)

    def test_constraints(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
          - 1
          - 2
          - 3
          - 4
      objectives:
        - perf_throughput
        - gpu_used_memory
      constraints:
        gpu_used_memory:
          max: 80
  - vgg_19_graphdef
"""
        config = self._evaluate_config(args, yaml_content)
        expected_model_objects = [
            ConfigModel('vgg_16_graphdef',
                        parameters={'concurrency': [1, 2, 3, 4]},
                        objectives=['perf_throughput', 'gpu_used_memory'],
                        constraints={'gpu_used_memory': {
                            'max': 80,
                        }}),
            ConfigModel('vgg_19_graphdef')
        ]
        self._assert_equality_of_model_configs(
            config.get_all_config()['model_names'], expected_model_objects)

        # GPU Memory shouldn't have min
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
          - 1
          - 2
          - 3
          - 4
      objectives:
        - perf_throughput
        - gpu_used_memory
      constraints:
        gpu_memory:
          max: 80
          min: 45
  - vgg_19_graphdef
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)

        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()
        mock_config.stop()

    def test_validation(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]

        # end key should not be included in concurrency
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
            start: 4
            stop: 12
            end: 2
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)

        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()
        mock_config.stop()

        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]

        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
            start: 13
            stop: 12
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)

        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()
        mock_config.stop()

        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
      parameters:
        concurrency: []
"""
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = AnalyzerConfig()
        cli = CLI(config)

        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()
        mock_config.stop()

    def test_config_model(self):
        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            instance_group:
                -
                    kind: KIND_GPU
                    count: 1

"""
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()['model_names']
        expected_model_configs = [
            ConfigModel('vgg_16_graphdef',
                        model_config_parameters={
                            'instance_group': [{
                                'kind': 'KIND_GPU',
                                'count': 1
                            }]
                        })
        ]
        self._assert_equality_of_model_configs(model_configs,
                                               expected_model_configs)

        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            instance_group:
                -
                    kind: KIND_GPU
                    count: 1
                -
                    kind: KIND_CPU
                    count: 1

"""
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()['model_names']
        expected_model_configs = [
            ConfigModel('vgg_16_graphdef',
                        model_config_parameters={
                            'instance_group': [{
                                'kind': 'KIND_GPU',
                                'count': 1
                            }, {
                                'kind': 'KIND_CPU',
                                'count': 1
                            }]
                        })
        ]
        self._assert_equality_of_model_configs(model_configs,
                                               expected_model_configs)

        args = [
            'model-analyzer', '--model-repository', 'cli_repository', '-f',
            'path-to-config-file'
        ]
        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            instance_group:
            -
                -
                    kind: KIND_GPU
                    count: 1
            -
                -
                    kind: KIND_CPU
                    count: 1

"""
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()['model_names']
        expected_model_configs = [
            ConfigModel('vgg_16_graphdef',
                        model_config_parameters={
                            'instance_group': [[{
                                'kind': 'KIND_GPU',
                                'count': 1
                            }], [{
                                'kind': 'KIND_CPU',
                                'count': 1
                            }]]
                        })
        ]
        self._assert_equality_of_model_configs(model_configs,
                                               expected_model_configs)

        yaml_content = """
model_names:
  -
    vgg_16_graphdef:
        model_config_parameters:
            input:
              -
                name: NV_MODEL_INPUT
                data_type: TYPE_FP32
                format: FORMAT_NHWC
                dims: [256, 256, 3]

"""
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()['model_names']
        expected_model_configs = [
            ConfigModel('vgg_16_graphdef',
                        model_config_parameters={
                            'input': [{
                                'name': 'NV_MODEL_INPUT',
                                'data_type': 'TYPE_FP32',
                                'format': 'FORMAT_NHWC',
                                'dims': [256, 256, 3]
                            }]
                        })
        ]
        model_config = config.get_config()['model_names']
        self._assert_model_config_types(model_config)

        model = model_config.field_type().raw_value().raw_value()[0]
        self._assert_model_object_types(model, 'vgg_16_graphdef')

        model_config_parameters = model.raw_value().raw_value(
        )['vgg_16_graphdef'].raw_value()['model_config_parameters']

        self._assert_model_config_params(model_config_parameters)
        self._assert_equality_of_model_configs(model_configs,
                                               expected_model_configs)


if __name__ == '__main__':
    unittest.main()
