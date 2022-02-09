# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
                            '--profile-models',
                            type=str,
                            required=True,
                            help='The config file for this test')

        args = parser.parse_args()
        self.config = {'batch_sizes': [1, 2], 'concurrency': [1, 2]}
        self.config['profile_models'] = sorted(args.profile_models.split(','))
        self.config['run_config_search_disable'] = True
        # Triton Server ssl flags
        self.config['grpc-use-ssl'] = '1'
        self.config['grpc-use-ssl-mutual'] = '1'
        self.config['grpc-server-cert'] = './server.crt'
        self.config['grpc-server-key'] = './server.key'
        self.config['grpc-root-cert'] = './ca.crt'
        # Perf Analyzer ssl flags
        self.config['ssl-grpc-use-ssl'] = True
        self.config['ssl-grpc-root-certifications-file'] = './ca.crt'
        self.config['ssl-grpc-private-key-file'] = './client.key'
        self.config['ssl-grpc-certificate-chain-file'] = './client.crt'

    def generate_config(self):
        with open('config.yml', 'w+') as f:
            yaml.dump(self.config, f)


if __name__ == '__main__':
    TestConfigGenerator()
