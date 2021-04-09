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
        self.config = {}
        self.config['model_names'] = {
            model_name: {
                'parameters': {
                    'concurrency': [1, 4],
                    'batch_sizes': [8, 16]
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

    def generate_summarize(self):
        self.config['summarize'] = True
        self.config['num_top_model_configs'] = 2
        with open('config-summarize.yml', 'w+') as f:
            yaml.dump(self.config, f)


if __name__ == '__main__':
    TestConfigGenerator()
