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
        self.test_id = -1
        test_functions = [
            self.__getattribute__(name)
            for name in dir(self)
            if name.startswith("generate")
        ]
        for test_function in test_functions:
            self.setUp()
            test_function()

    def setUp(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m",
                            "--profile-models",
                            type=str,
                            required=True,
                            help="The models to be profiled for this test")
        args = parser.parse_args()
        self.profile_models = args.profile_models.split(",")
        self.test_id += 1

    def generate_search_disable(self):
        concurrency_count = 1  # 1 element in concurrency below
        instance_count = 2  # 2 elements in instance_group below
        model_config = {
            "run_config_search_disable": True,
            "profile_models": {
                model: {
                    "parameters": {
                        "concurrency": [1]
                    },
                    "model_config_parameters": {
                        "instance_group": [{
                            "count": [1, 2],
                            "kind": "KIND_GPU"
                        }]
                    }
                } for model in self.profile_models
            }
        }
        total_param_count = self._calculate_total_params(
            concurrency_count, instance_count)
        self._write_file(total_param_count, 1, 2, 1, model_config)

    def generate_max_limit_with_model_config(self):
        concurrency_count = 2
        instance_count = 2
        model_config = {
            "run_config_search_max_concurrency": concurrency_count,
            "run_config_search_max_instance_count": instance_count,
            "profile_models": {
                model: {
                    "model_config_parameters": {
                        "instance_group": [{
                            "count": [1, 2],
                            "kind": "KIND_GPU"
                        }]
                    }
                } for model in self.profile_models
            },
        }
        total_param_count = self._calculate_total_params(
            concurrency_count, instance_count)
        self._write_file(total_param_count, 2, 2, 1, model_config)

    def generate_max_limit(self):
        concurrency_count = 2
        instance_count = 2
        model_config = {
            "run_config_search_max_concurrency": concurrency_count,
            "run_config_search_max_instance_count": instance_count,
            "profile_models": self.profile_models,
        }
        total_param_count = self._calculate_total_params(
            concurrency_count, instance_count)
        self._write_file(total_param_count, 2, 8, 1, model_config)

    def generate_max_limit_with_param(self):
        concurrency_count = 1  # 1 because concurrency parameter is 1 entry below
        instance_count = 2
        model_config = {
            "run_config_search_max_concurrency": 2,
            "run_config_search_max_instance_count": 2,
            "profile_models": {
                model: {
                    "parameters": {
                        "concurrency": [1]
                    },
                } for model in self.profile_models
            },
        }
        total_param_count = self._calculate_total_params(
            concurrency_count, instance_count)
        self._write_file(total_param_count, 1, 6, 1, model_config)

    def generate_max_limit_with_param_and_model_config(self):
        concurrency_count = 1  # 1 because concurrency parameter is 1 entry below
        instance_count = 2
        model_config = {
            "run_config_search_max_concurrency": 2,
            "run_config_search_max_instance_count": 2,
            "profile_models": {
                model: {
                    "parameters": {
                        "concurrency": [1]
                    },
                    "model_config_parameters": {
                        "instance_group": [{
                            "count": [1, 2],
                            "kind": "KIND_GPU"
                        }]
                    }
                } for model in self.profile_models
            },
        }
        total_param_count = self._calculate_total_params(
            concurrency_count, instance_count)
        self._write_file(total_param_count, 1, 2, 1, model_config)

    def generate_max_limit_with_dynamic_batch_disable(self):
        model_config = {
            "run_config_search_max_concurrency": 2,
            "run_config_search_max_instance_count": 2,
            "profile_models": self.profile_models,
        }
        self._write_file(6, 2, 4, 1, model_config)

    def _calculate_total_params(self,
                                concurrency,
                                instance_count,
                                default_config_count=1):
        concurrency_count = len(range(1, concurrency))
        instance_count = len(range(1, instance_count))
        return concurrency_count * (instance_count + default_config_count)

    def _write_file(self, total_param, total_param_remote, total_models,
                    total_models_remote, model_config):
        with open(f"./config-{self.test_id}-param.txt", "w") as file:
            file.write(str(total_param))
        with open(f"./config-{self.test_id}-param-remote.txt", "w") as file:
            file.write(str(total_param_remote))
        with open(f"./config-{self.test_id}-models.txt", "w") as file:
            file.write(str(total_models))
        with open(f"./config-{self.test_id}-models-remote.txt", "w") as file:
            file.write(str(total_models_remote))
        with open(f"./config-{self.test_id}.yml", "w") as file:
            yaml.dump(model_config, file)


if __name__ == '__main__':
    TestConfigGenerator()
