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
import os


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
        parser.add_argument(
            '-p',
            '--protocols',
            type=str,
            required=True,
            help='Comma separated list of client protocols for this test')
        parser.add_argument(
            '-l',
            '--launch-modes',
            type=str,
            required=True,
            help='Comma separated list of launch modes for this test')

        self.args = parser.parse_args()
        self.protocols = sorted(self.args.protocols.split(','))
        self.launch_modes = sorted(self.args.launch_modes.split(','))

        self.config = {}
        self.config['run_config_search_disable'] = True
        self.config['batch_sizes'] = 4
        self.config['concurrency'] = 4
        self.config['perf_analyzer_cpu_util'] = 6000

    def generate_configs(self):
        for launch_mode in self.launch_modes:
            self.config['triton_launch_mode'] = launch_mode

            if launch_mode == 'c_api':
                self.config['perf_output'] = True
                with open(f'config-{launch_mode}-c_api.yaml', 'w') as f:
                    yaml.dump(self.config, f)
            else:
                self.config['perf_output'] = False
                for protocol in self.protocols:
                    self.config['client_protocol'] = protocol
                    if launch_mode == 'docker':
                        # Set docker image and put in the CI runner's labels
                        if 'TRITON_LAUNCH_DOCKER_IMAGE' in os.environ:
                            self.config['triton_docker_image'] = os.environ[
                                'TRITON_LAUNCH_DOCKER_IMAGE']
                        if 'RUNNER_ID' in os.environ:
                            self.config['triton_docker_labels'] = {
                                'RUNNER_ID': os.environ['RUNNER_ID']
                            }
                    with open(f'config-{launch_mode}-{protocol}.yaml',
                              'w') as f:
                        yaml.dump(self.config, f)


if __name__ == '__main__':
    TestConfigGenerator()
