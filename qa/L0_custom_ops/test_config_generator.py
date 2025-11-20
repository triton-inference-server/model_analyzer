#!/usr/bin/env python3

# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

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
            if name.startswith("generate")
        ]

        for test_function in test_functions:
            self.setup()
            test_function()

    def setup(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m",
            "--profile-models",
            type=str,
            required=True,
            help="The config file for this test",
        )
        parser.add_argument(
            "-p",
            "--preload-path",
            type=str,
            required=True,
            help="The path to the custom op shared object",
        )
        parser.add_argument(
            "-l",
            "--library-path",
            type=str,
            required=True,
            help="The path to the backend shared libraries used by the custom op",
        )
        parser.add_argument(
            "--docker-model-repository",
            type=str,
            required=True,
            help="The model repository path accessible to Docker daemon (for Docker mode)",
        )
        parser.add_argument(
            "--local-model-repository",
            type=str,
            required=True,
            help="The local copy of model repository (for c_api and local modes)",
        )

        self.args = parser.parse_args()
        self.profile_models = sorted(self.args.profile_models.split(","))

        self.config = {}
        self.config["run_config_search_disable"] = True

        # Multiple model configs to confirm we don't lose any information
        # between launches of triton server
        self.config["profile_models"] = {
            model: {
                "model_config_parameters": {
                    "instance_group": [{"count": [1, 2], "kind": "KIND_CPU"}]
                }
            }
            for model in self.profile_models
        }
        self.config["batch_sizes"] = 1
        self.config["concurrency"] = 1

    def generate_local_mode_custom_op_config(self):
        self.config["triton_launch_mode"] = "local"
        # For local mode, use the local copy's custom_modulo.so path
        preload_parts = self.args.preload_path.split(":")
        local_preload = f"{preload_parts[0]}:{self.args.local_model_repository}/libtorch_modulo/custom_modulo.so"
        self.config["triton_server_environment"] = {
            "LD_PRELOAD": local_preload,
            "LD_LIBRARY_PATH": self.args.library_path,
        }
        with open("config-local.yaml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_docker_mode_custom_op_config(self):
        self.config["triton_launch_mode"] = "docker"
        if "TRITON_LAUNCH_DOCKER_IMAGE" in os.environ:
            self.config["triton_docker_image"] = os.environ[
                "TRITON_LAUNCH_DOCKER_IMAGE"
            ]
        # Mount the Docker-accessible model repository path
        # Use /tmp/output path which is accessible to Docker daemon on host
        self.config["triton_docker_mounts"] = [
            f"{self.args.docker_model_repository}:{self.args.docker_model_repository}:ro"
        ]
        # For Docker mode, use the docker repository path for LD_PRELOAD
        docker_preload = f"{self.args.preload_path}:{self.args.docker_model_repository}/libtorch_modulo/custom_modulo.so"
        self.config["triton_server_environment"] = {
            "LD_PRELOAD": docker_preload,
            "LD_LIBRARY_PATH": self.args.library_path,
        }
        with open("config-docker.yaml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_c_api_custom_op_config(self):
        self.config["triton_launch_mode"] = "c_api"
        self.config["perf_output"] = True
        # For c_api mode, use the local copy's custom_modulo.so path
        preload_parts = self.args.preload_path.split(":")
        local_preload = f"{preload_parts[0]}:{self.args.local_model_repository}/libtorch_modulo/custom_modulo.so"
        self.config["triton_server_environment"] = {
            "LD_PRELOAD": local_preload,
            "LD_LIBRARY_PATH": self.args.library_path,
        }
        with open("config-c_api.yaml", "w+") as f:
            yaml.dump(self.config, f)


if __name__ == "__main__":
    TestConfigGenerator()
