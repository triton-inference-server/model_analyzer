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


class RunConfig:
    """
    Encapsulates all the information needed to run one or more models 
    at the same time in Perf Analyzer
    """

    def __init__(self, triton_env):
        """
        Parameters
        ----------
        triton_env : dict
            A dictionary of environment variables to set
            when launching tritonserver
        """

        self._triton_env = triton_env
        self._model_run_configs = []

    def add_model_run_config(self, model_run_config):
        """
        Add a ModelRunConfig to this RunConfig
        """
        self._model_run_configs.append(model_run_config)

    def model_run_configs(self):
        """
        Returns the list of ModelRunConfigs to run concurrently
        """
        return self._model_run_configs

    def is_legal_combination(self):
        """
        Returns true if all model_run_configs are valid
        """
        return all([
            model_run_config.is_legal_combination()
            for model_run_config in self._model_run_configs
        ])

    def triton_environment(self):
        """
        Returns
        -------
        dict
            The environment that tritonserver
            was run with for this RunConfig
        """

        return self._triton_env
