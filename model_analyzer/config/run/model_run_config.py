# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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


class ModelRunConfig:
    """
    A class that encapsulates all the information needed to run a model in Perf Analyzer
    """

    def __init__(self, model_name, model_config, perf_config, triton_env):
        """
        Parameters
        ----------
        model_name: str
            The name of the model
        model_config : ModelConfig
            model config corresponding to this run
        perf_config : PerfAnalyzerConfig
            List of possible run parameters to pass
            to Perf Analyzer
        FIXME triton_env goes away
        triton_env : dict
            A dictionary of environment variables to set
            when launching tritonserver
        """

        self._model_name = model_name
        self._model_config = model_config
        self._perf_config = perf_config
        self._triton_env = triton_env

    def model_name(self):
        """
        Get the original model name for this run config.

        Returns
        -------
        str
            Original model name
        """

        return self._model_name

    def model_config(self):
        """
        Returns
        -------
        List of ModelConfig
            The list of ModelConfigs corresponding to this run.
        """

        return self._model_config

    def perf_config(self):
        """
        Returns
        -------
        PerfAnalyzerConfig
            run parameters corresponding to this run of 
            the perf analyzer
        """

        return self._perf_config

    def is_legal_combination(self):
        """
        Returns true if the run_config is valid and should be run. Else false
        """
        model_config = self._model_config.get_config()

        max_batch_size = model_config[
            'max_batch_size'] if 'max_batch_size' in model_config else 1
        perf_batch_size = self._perf_config[
            'batch-size'] if 'batch-size' in self._perf_config else 1

        return max_batch_size >= perf_batch_size

    def triton_environment(self):
        """
        Returns
        -------
        dict
            The environment that tritonserver
            was run with for this ModelRunConfig
        """

        return self._triton_env
