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
        self._write_file(2, 1, 2, 1, model_config)

    def generate_max_limit_with_model_config(self):
        model_config = {
            "run_config_search_max_concurrency": 2,
            "run_config_search_max_instance_count": 2,
            "run_config_search_max_preferred_batch_size": 2,
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
        self._write_file(4, 2, 2, 1, model_config)

    def generate_max_limit(self):
        model_config = {
            "run_config_search_max_concurrency": 2,
            "run_config_search_max_instance_count": 2,
            "run_config_search_max_preferred_batch_size": 2,
            "profile_models": self.profile_models,
        }
        self._write_file(16, 2, 8, 1, model_config)

    def generate_max_limit_with_param(self):
        model_config = {
            "run_config_search_max_concurrency": 2,
            "run_config_search_max_instance_count": 2,
            "run_config_search_max_preferred_batch_size": 1,
            "profile_models": {
                model: {
                    "parameters": {
                        "concurrency": [1]
                    },
                } for model in self.profile_models
            },
        }
        self._write_file(6, 1, 6, 1, model_config)

    def generate_max_limit_with_param_and_model_config(self):
        model_config = {
            "run_config_search_max_concurrency": 2,
            "run_config_search_max_instance_count": 2,
            "run_config_search_max_preferred_batch_size": 2,
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
        self._write_file(2, 1, 2, 1, model_config)

    def generate_max_limit_with_dynamic_batch_disable(self):
        model_config = {
            "run_config_search_max_concurrency": 2,
            "run_config_search_max_instance_count": 2,
            "run_config_search_preferred_batch_size_disable": True,
            "profile_models": self.profile_models,
        }
        self._write_file(4, 2, 4, 1, model_config)

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
