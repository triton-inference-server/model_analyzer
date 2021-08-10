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

    def generate_with_cpu_metrics(self):
        model_config = {
            "collect_cpu_metrics":
                True,
            "run_config_search_disable":
                True,
            "profile_models": {
                model: {
                    "parameters": {
                        "concurrency": [1]
                    },
                    "model_config_parameters": {
                        "instance_group": [{
                            "count": [1],
                            "kind": "KIND_GPU"
                        }],
                        "dynamic_batching": [{
                            "preferred_batch_size": [[1]]
                        }]
                    }
                } for model in self.profile_models
            },
            "inference_output_fields": [
                "model_name", "batch_size", "concurrency", "model_config_path",
                "instance_group", "dynamic_batch_sizes",
                "satisfies_constraints", "perf_throughput", "perf_latency",
                "cpu_used_ram"
            ]
        }
        self._write_file(5, 11, 10, model_config)

    def generate_without_cpu_metrics(self):
        model_config = {
            "collect_cpu_metrics": False,
            "run_config_search_disable": True,
            "profile_models": {
                model: {
                    "parameters": {
                        "concurrency": [1]
                    },
                    "model_config_parameters": {
                        "instance_group": [{
                            "count": [1],
                            "kind": "KIND_GPU"
                        }],
                        "dynamic_batching": [{
                            "preferred_batch_size": [[1]]
                        }]
                    }
                } for model in self.profile_models
            }
        }
        self._write_file(5, 11, 9, model_config)

    def _write_file(self, total_param_server, total_param_gpu,
                    total_param_inference, model_config):
        with open(f"./config-{self.test_id}-param-server.txt", "w") as file:
            file.write(str(total_param_server))
        with open(f"./config-{self.test_id}-param-gpu.txt", "w") as file:
            file.write(str(total_param_gpu))
        with open(f"./config-{self.test_id}-param-inference.txt", "w") as file:
            file.write(str(total_param_inference))
        with open(f"./config-{self.test_id}.yml", "w") as file:
            yaml.dump(model_config, file)


if __name__ == "__main__":
    TestConfigGenerator()
