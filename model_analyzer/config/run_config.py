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


class RunConfig:
    """
    A class that encapsulates all the information
    needed to complete a single run of the Model
    Analyzer. One run corresponds to
    one ModelConfig and produces one RunResult. 
    """

    def __init__(self, model_config, perf_configs):
        """
        Parameters
        ----------
        model_config : ModelConfig
            The model config corresponding to this run
        perf_configs : list of PerfAnalyzerConfigs
            list of possible run parameters to pass
            to Perf Analyzer
        """

        self._model_config = model_config
        self._perf_analyzer_configs = perf_configs

    def model_config(self):
        """
        Returns
        -------
        ModelConfig
            The ModelConfig corresponding to this run. 
        """

        return self._model_config

    def perf_analyzer_configs(self):
        """
        Returns
        -------
        List of PerfAnalyzerConfig objects
            This list of run parameters corresponding to this run
        """

        return self._perf_analyzer_configs
