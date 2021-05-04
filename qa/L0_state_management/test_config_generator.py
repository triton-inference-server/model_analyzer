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
    
    The `setup` function does the work common to all tests

    TO ADD A TEST: Simply add a member function whose name starts
                    with 'generate'.
    """
    def __init__(self):
        test_functions = [
            self.__getattribute__(name) for name in dir(self)
            if name.startswith('generate')
        ]

        for test_function in test_functions:
            self.setup()
            test_function()

    def setup(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-m',
                            '--model-names',
                            type=str,
                            required=True,
                            help='The config file for this test')

        args = parser.parse_args()
        self.model_names = sorted(args.model_names.split(','))
        self.config = {'batch_sizes': 1, 'num_top_model_configs': 3}
        self.config['model_names'] = {
            model_name: {
                'parameters': {
                    'concurrency': [16],
                },
                'model_config_parameters': {
                    'instance_group': [{
                        'kind': 'KIND_GPU',
                        'count': 1
                    }]
                }
            }
            for model_name in self.model_names
        }

    def generate_config_single(self):
        with open('config-single.yml', 'w+') as f:
            yaml.dump(self.config, f)

    def generate_config_multi(self):
        self.config['model_names'][
            self.model_names[1]]['parameters']['concurrency'] = [16, 32]
        with open('config-multi.yml', 'w+') as f:
            yaml.dump(self.config, f)

    def generate_config_mixed_first(self):
        """
        Generate a config where last two models
        are removed
        """

        for model_name in self.model_names[-1:]:
            del self.config['model_names'][model_name]
        with open('config-mixed-first.yml', 'w+') as f:
            yaml.dump(self.config, f)

    def generate_config_mixed_second(self):
        """
        Generate config where first two models are
        removed, and 3rd model has changed run parameters
        """
        for model_name in self.model_names[:1]:
            del self.config['model_names'][model_name]
        self.config['model_names'][
            self.model_names[1]]['parameters']['concurrency'] = [32]
        with open('config-mixed-second.yml', 'w+') as f:
            yaml.dump(self.config, f)


if __name__ == '__main__':
    TestConfigGenerator()
