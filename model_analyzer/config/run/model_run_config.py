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

from typing import List, Optional

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

    def __init__(self, model_name: str, model_config: ModelConfig,
                 perf_config: PerfAnalyzerConfig) -> None:
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
        self._ensemble_subconfigs: List[ModelConfig] = []

    def model_name(self) -> str:
        """
        Get the original model name for this run config.

        Returns
        -------
        str
            Original model name
        """

        return self._model_name

    def model_variant_name(self) -> str:
        """
        Get the model config variant name for this config.

        Returns
        -------
        str
            Model variant name
        """

        return self.model_config().get_field('name')

    def model_config(self) -> ModelConfig:
        """
        Returns
        -------
        List of ModelConfig
            The list of ModelConfigs corresponding to this run.
        """

        return self._model_config

    def perf_config(self) -> PerfAnalyzerConfig:
        """
        Returns
        -------
        PerfAnalyzerConfig
            run parameters corresponding to this run of 
            the perf analyzer
        """

        return self._perf_config

    def ensemble_subconfigs(self) -> List[ModelConfig]:
        """
        Returns the list of ensemble subconfigs
        """

        return self._ensemble_subconfigs

    def representation(self) -> str:
        """
        Returns a representation string for the ModelRunConfig that can be used
        as a key to uniquely identify it
        """

        return self.perf_config().representation()

    def _check_for_client_vs_model_batch_size(self):
        """
        Returns false if client batch size is greater than model batch size. Else true
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

    def _check_for_preferred_vs_model_batch_size(self):
        """
        Returns false if maximum of preferred batch size is greater than model batch size. Else true
        """
        legal = True
        ensemble_subconfigs = [
            subconfig.get_config() for subconfig in self._ensemble_subconfigs
        ]
        model_configs = ensemble_subconfigs if self._ensemble_subconfigs else [
            self._model_config.get_config()
        ]

        for model_config in model_configs:
            max_batch_size = model_config[
                'max_batch_size'] if 'max_batch_size' in model_config else self.DEFAULT_MAX_BATCH_SIZE

            if 'dynamic_batching' in model_config and 'preferred_batch_size' in model_config[
                    'dynamic_batching']:
                max_preferred_batch_size = max(
                    model_config['dynamic_batching']['preferred_batch_size'])
                legal = max_batch_size >= max_preferred_batch_size

                if not legal:
                    logger.debug(
                        f"Illegal model run config because maximum of {model_config['name']}'s preferred batch size {max_preferred_batch_size} is greater than model max batch size {max_batch_size}"
                    )
                    return legal

        return legal

    def is_legal_combination(self):
        """
        Returns true if the run_config is valid and should be run. Else false
        """
        legal = self._check_for_client_vs_model_batch_size() and \
            self._check_for_preferred_vs_model_batch_size()

        return legal

    def is_ensemble_model(self) -> bool:
        """
        Returns true if the model_config is an ensemble model
        """
        return self._model_config.is_ensemble()

    def get_ensemble_subconfig_names(self) -> Optional[List[str]]:
        """
        Returns list of Ensemble Subconfig names
        """
        return self._model_config.get_ensemble_submodels(
        ) if self._model_config.is_ensemble() else []

    def add_ensemble_submodel_configs(
            self, submodel_configs: List[ModelConfig]) -> None:
        """
        Adds a list of ensemble submodel configs
        """
        for submodel_config in submodel_configs:
            self._ensemble_subconfigs.append(submodel_config)

    @classmethod
    def from_dict(cls, model_run_config_dict):
        model_run_config = ModelRunConfig(None, None, None)
        model_run_config._model_name = model_run_config_dict['_model_name']
        model_run_config._model_config = ModelConfig.from_dict(
            model_run_config_dict['_model_config'])
        model_run_config._perf_config = PerfAnalyzerConfig.from_dict(
            model_run_config_dict['_perf_config'])

        if '_ensemble_subconfigs' in model_run_config_dict:
            model_run_config._ensemble_subconfigs = [
                ModelConfig.from_dict(ensemble_subconfig_dict)
                for ensemble_subconfig_dict in
                model_run_config_dict['_ensemble_subconfigs']
            ]

        return model_run_config
