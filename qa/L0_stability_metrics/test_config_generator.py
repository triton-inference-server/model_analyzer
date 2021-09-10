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
            self.__getattribute__(name)
            for name in dir(self)
            if name.startswith('generate')
        ]

        for test_function in test_functions:
            self.setup()
            test_function()

    def setup(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-m',
                            '--models',
                            type=str,
                            required=True,
                            help='The models used for this test')

        self.args = parser.parse_args()
        self.models = sorted(self.args.models.split(','))

        self.config = {}

        # Profile config
        self.config['run_config_search_disable'] = True
        self.config['concurrency'] = 16
        self.config['batch-size'] = 8
        self.config['profile_models'] = self.models

        # Analyze config
        self.config['summarize'] = False
        self.config['collect_cpu_metrics'] = True
        self.config['gpu_output_fields'] = [
            'model_name', 'batch_size', 'concurrency', 'gpu_used_memory',
            'gpu_utilization'
        ]
        self.config['analysis_models'] = {}
        for model in self.models:
            self.config['analysis_models'][model] = {
                'objectives': {
                    'perf_throughput': 10
                }
            }

    def generate_configs(self):
        with open('config.yaml', 'w+') as f:
            yaml.dump(self.config, f)


if __name__ == '__main__':
    TestConfigGenerator()
