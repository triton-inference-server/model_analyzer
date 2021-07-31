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
                            help="The config file for this test")
        args = parser.parse_args()
        self.profile_models = args.profile_models.split(",")

        self.config = {}
        self.config["run_config_search_max_concurrency"] = 1
        self.config["run_config_search_max_instance_count"] = 1
        self.config["run_config_search_max_preferred_batch_size"] = 1

    def generate_gpu_on_cpu_only_on_monitor_auto(self):
        """
        Check: 
            GPU visible: True
            CPU only: True
            CPU monitor disable: auto
        CPU monitor should run: True
        """
        self.config["gpus"] = "all"
        self.config["cpu_monitor_disable"] = "auto"
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": True
            } for model_name in self.profile_models
        }
        with open("config-gpu-on-cpu-only-on-monitor-auto.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_gpu_on_cpu_only_on_monitor_off(self):
        """
        Check: 
            GPU visible: True
            CPU only: True
            CPU monitor disable: True
        CPU monitor should run: False
        """
        self.config["gpus"] = "all"
        self.config["cpu_monitor_disable"] = True
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": True
            } for model_name in self.profile_models
        }
        with open("config-gpu-on-cpu-only-on-monitor-off.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_gpu_on_cpu_only_on_monitor_on(self):
        """
        Check: 
            GPU visible: True
            CPU only: True
            CPU monitor disable: False
        CPU monitor should run: True
        """
        self.config["gpus"] = "all"
        self.config["cpu_monitor_disable"] = False
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": True
            } for model_name in self.profile_models
        }
        with open("config-gpu-on-cpu-only-on-monitor-on.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_gpu_on_cpu_only_off_monitor_auto(self):
        """
        Check: 
            GPU visible: True
            CPU only: False
            CPU monitor disable: auto
        CPU monitor should run: False
        """
        self.config["gpus"] = "all"
        self.config["cpu_monitor_disable"] = "auto"
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": False
            } for model_name in self.profile_models
        }
        with open("config-gpu-on-cpu-only-off-monitor-auto.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_gpu_on_cpu_only_off_monitor_off(self):
        """
        Check: 
            GPU visible: True
            CPU only: False
            CPU monitor disable: True
        CPU monitor should run: False
        """
        self.config["gpus"] = "all"
        self.config["cpu_monitor_disable"] = True
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": False
            } for model_name in self.profile_models
        }
        with open("config-gpu-on-cpu-only-off-monitor-off.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_gpu_on_cpu_only_off_monitor_on(self):
        """
        Check: 
            GPU visible: True
            CPU only: False
            CPU monitor disable: False
        CPU monitor should run: True
        """
        self.config["gpus"] = "all"
        self.config["cpu_monitor_disable"] = False
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": False
            } for model_name in self.profile_models
        }
        with open("config-gpu-on-cpu-only-off-monitor-on.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_gpu_off_cpu_only_on_monitor_auto(self):
        """
        Check: 
            GPU visible: False
            CPU only: True
            CPU monitor disable: auto
        CPU monitor should run: True
        """
        self.config["gpus"] = []
        self.config["cpu_monitor_disable"] = "auto"
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": True
            } for model_name in self.profile_models
        }
        with open("config-gpu-off-cpu-only-on-monitor-auto.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_gpu_off_cpu_only_on_monitor_off(self):
        """
        Check: 
            GPU visible: False
            CPU only: True
            CPU monitor disable: True
        CPU monitor should run: False
        """
        self.config["gpus"] = []
        self.config["cpu_monitor_disable"] = True
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": True
            } for model_name in self.profile_models
        }
        with open("config-gpu-off-cpu-only-on-monitor-off.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_gpu_off_cpu_only_on_monitor_on(self):
        """
        Check: 
            GPU visible: False
            CPU only: True
            CPU monitor disable: False
        CPU monitor should run: True
        """
        self.config["gpus"] = []
        self.config["cpu_monitor_disable"] = False
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": True
            } for model_name in self.profile_models
        }
        with open("config-gpu-off-cpu-only-on-monitor-on.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_gpu_off_cpu_only_off_monitor_auto(self):
        """
        Check: 
            GPU visible: False
            CPU only: False
            CPU monitor disable: auto
        CPU monitor should run: True
        """
        self.config["gpus"] = []
        self.config["cpu_monitor_disable"] = "auto"
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": False
            } for model_name in self.profile_models
        }
        with open("config-gpu-off-cpu-only-off-monitor-auto.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_gpu_off_cpu_only_off_monitor_off(self):
        """
        Check: 
            GPU visible: False
            CPU only: False
            CPU monitor disable: True
        CPU monitor should run: False
        """
        self.config["gpus"] = []
        self.config["cpu_monitor_disable"] = True
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": False
            } for model_name in self.profile_models
        }
        with open("config-gpu-off-cpu-only-off-monitor-off.yml", "w+") as f:
            yaml.dump(self.config, f)

    def generate_gpu_off_cpu_only_off_monitor_on(self):
        """
        Check: 
            GPU visible: False
            CPU only: False
            CPU monitor disable: False
        CPU monitor should run: True
        """
        self.config["gpus"] = []
        self.config["cpu_monitor_disable"] = False
        self.config["profile_models"] = {
            model_name: {
                "cpu_only": False
            } for model_name in self.profile_models
        }
        with open("config-gpu-off-cpu-only-off-monitor-on.yml", "w+") as f:
            yaml.dump(self.config, f)


if __name__ == "__main__":
    TestConfigGenerator()
