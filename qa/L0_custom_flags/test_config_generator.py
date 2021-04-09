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

import argparse
import yaml


class TestConfigGenerator:
    """
    This class contains functions that
    create configs for various test scenarios.

    TO ADD A TEST: Simply add a member function whose name starts
                    with 'generate'.
    """
    def __init__(self):
        test_functions = [
            self.__getattribute__(name) for name in dir(self)
            if name.startswith('generate')
        ]

        for test_function in test_functions:
            self.setUp()
            test_function()

    def setUp(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-m',
                            '--model-names',
                            type=str,
                            required=True,
                            help='The config file for this test')

        args = parser.parse_args()
        model_names = args.model_names.split(',')
        self.config = {'perf_output': True}
        self.config['model_names'] = {
            model_name: {
                'parameters': {
                    'concurrency': [1],
                    'batch_sizes': [1]
                },
                'model_config_parameters': {
                    'instance_group': [{
                        'kind': 'KIND_GPU',
                        'count': 1
                    }]
                }
            }
            for model_name in model_names
        }

    def generate_perf_flags_per_model(self):
        for i, model_name in enumerate(self.config['model_names']):
            self.config['model_names'][model_name]['perf_analyzer_flags'] = {
                'percentile': 95 + i
            }
        with open('config-perf-per-model.yml', 'w+') as f:
            yaml.dump(self.config, f)

    def generate_perf_flags_global_autofill(self):
        self.config['perf_analyzer_flags'] = {'percentile': 50}
        with open('config-perf-global.yml', 'w+') as f:
            yaml.dump(self.config, f)

    def generate_triton_flags_per_model(self):
        for i, model_name in enumerate(self.config['model_names']):
            self.config['model_names'][model_name]['triton_server_flags'] = {
                'exit-timeout-secs': 100 + i
            }
        with open('config-triton-per-model.yml', 'w+') as f:
            yaml.dump(self.config, f)

    def generate_triton_flags_global(self):
        self.config['triton_server_flags'] = {'strict-model-config': False}
        with open('config-triton-global.yml', 'w+') as f:
            yaml.dump(self.config, f)


if __name__ == '__main__':
    TestConfigGenerator()
