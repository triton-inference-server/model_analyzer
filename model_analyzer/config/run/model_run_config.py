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

from model_analyzer.constants import LOGGER_NAME
import logging

from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig

logger = logging.getLogger(LOGGER_NAME)


class ModelRunConfig:
    """
    Encapsulates all the information (ModelConfig + PerfConfig) needed to run
    a model in Perf Analyzer
    """
    DEFAULT_MAX_BATCH_SIZE = 1
    DEFAULT_PERF_BATCH_SIZE = 1

    def __init__(self, model_name, model_config, perf_config):
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
        """

        self._model_name = model_name
        self._model_config = model_config
        self._perf_config = perf_config

    def model_name(self):
        """
        Get the original model name for this run config.

        Returns
        -------
        str
            Original model name
        """

        return self._model_name

    def model_variant_name(self):
        """
        Get the model config variant name for this config.

        Returns
        -------
        str
            Model variant name
        """

        return self.model_config().get_field('name')

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

    def representation(self):
        """
        Returns a representation string for the ModelRunConfig that can be used
        as a key to uniquely identify it
        """
        return self.perf_config().representation()

    def is_legal_combination(self):
        """
        Returns true if the run_config is valid and should be run. Else false
        """
        model_config = self._model_config.get_config()

        max_batch_size = model_config[
            'max_batch_size'] if 'max_batch_size' in model_config else self.DEFAULT_MAX_BATCH_SIZE
        perf_batch_size = self._perf_config[
            'batch-size'] if 'batch-size' in self._perf_config else self.DEFAULT_PERF_BATCH_SIZE

        legal = max_batch_size >= perf_batch_size
        if not legal:
            logger.debug(
                f"Illegal model run config because client batch size {perf_batch_size} is greater than model max batch size {max_batch_size}"
            )
        return legal

    @classmethod
    def from_dict(cls, model_run_config_dict):
        model_run_config = ModelRunConfig(None, None, None)
        model_run_config._model_name = model_run_config_dict['_model_name']
        model_run_config._model_config = ModelConfig.from_dict(
            model_run_config_dict['_model_config'])
        model_run_config._perf_config = PerfAnalyzerConfig.from_dict(
            model_run_config_dict['_perf_config'])
        return model_run_config
