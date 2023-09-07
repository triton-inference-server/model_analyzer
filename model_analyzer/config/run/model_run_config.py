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

import logging
from typing import Dict, List, Optional

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.triton.model.model_config_variant import ModelConfigVariant

logger = logging.getLogger(LOGGER_NAME)


class ModelRunConfig:
    """
    Encapsulates all the information (ModelConfigVariant + PerfConfig) needed to run
    a model in Perf Analyzer
    """

    DEFAULT_MAX_BATCH_SIZE = 1
    DEFAULT_PERF_BATCH_SIZE = 1

    def __init__(
        self,
        model_name: str,
        model_config_variant: ModelConfigVariant,
        perf_config: PerfAnalyzerConfig,
    ) -> None:
        """
        Parameters
        ----------
        model_name: str
            The name of the model
        model_config_variant : ModelConfigVariant
            model config variant corresponding to this run
        perf_config : PerfAnalyzerConfig
            List of possible run parameters to pass
            to Perf Analyzer
        """

        self._model_name = model_name
        self._model_config_variant = model_config_variant
        self._perf_config = perf_config
        self._composing_config_variants: List[ModelConfigVariant] = []

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
        return (
            self._model_config_variant.variant_name
            if self._model_config_variant
            else ""
        )

    def model_config_variant(self) -> ModelConfigVariant:
        """
        Returns
        -------
        ModelConfigVariant
            The ModelConfigVariant corresponding to this run
        """

        return self._model_config_variant

    def model_config(self) -> Optional[ModelConfig]:
        """
        Returns
        -------
        ModelConfig
            The ModelConfig corresponding to this run
        """

        return (
            self._model_config_variant.model_config
            if self._model_config_variant
            else None
        )

    def perf_config(self) -> PerfAnalyzerConfig:
        """
        Returns
        -------
        PerfAnalyzerConfig
            run parameters corresponding to this run of
            the perf analyzer
        """

        return self._perf_config

    def composing_config_variants(self) -> List[ModelConfigVariant]:
        """
        Returns the list of composing model config variants
        """

        return self._composing_config_variants

    def composing_configs(self) -> List[ModelConfig]:
        """
        Returns the list of composing model configs
        """

        if self._composing_config_variants:
            composing_configs = [
                composing_config_variant.model_config
                for composing_config_variant in self._composing_config_variants
            ]
            return composing_configs
        else:
            return []

    def representation(self) -> str:
        """
        Returns a representation string for the ModelRunConfig that can be used
        as a key to uniquely identify it
        """
        repr = self.model_variant_name()
        repr += " " + self.perf_config().representation()

        if self._composing_config_variants:
            repr += " " + (",").join(self.get_composing_config_names())  # type: ignore

        return repr

    def _check_for_client_vs_model_batch_size(self) -> bool:
        """
        Returns false if client batch size is greater than model batch size. Else true
        """
        model_config = self._model_config_variant.model_config.get_config()

        max_batch_size = (
            model_config["max_batch_size"]
            if "max_batch_size" in model_config
            else self.DEFAULT_MAX_BATCH_SIZE
        )
        perf_batch_size = (
            self._perf_config["batch-size"]
            if "batch-size" in self._perf_config
            else self.DEFAULT_PERF_BATCH_SIZE
        )

        legal = max_batch_size >= perf_batch_size
        if not legal:
            logger.debug(
                f"Illegal model run config because client batch size {perf_batch_size} is greater than model max batch size {max_batch_size}"
            )

        return legal

    def _check_for_preferred_vs_model_batch_size(self) -> bool:
        """
        Returns false if maximum of preferred batch size is greater than model batch size. Else true
        """
        legal = True

        model_configs = self._create_model_config_dicts()

        for model_config in model_configs:
            max_batch_size = (
                model_config["max_batch_size"]
                if "max_batch_size" in model_config
                else self.DEFAULT_MAX_BATCH_SIZE
            )

            if (
                "dynamic_batching" in model_config
                and "preferred_batch_size" in model_config["dynamic_batching"]
            ):
                max_preferred_batch_size = max(
                    model_config["dynamic_batching"]["preferred_batch_size"]
                )
                legal = max_batch_size >= max_preferred_batch_size

                if not legal:
                    logger.debug(
                        f"Illegal model run config because maximum of {model_config['name']}'s preferred batch size {max_preferred_batch_size} is greater than model max batch size {max_batch_size}"
                    )
                    return legal

        return legal

    def _create_model_config_dicts(self) -> List[Dict]:
        """
        Create a list of model config dictionaries for
        the given model + composing models
        """
        model_configs = (
            []
            if self.is_ensemble_model()
            else [self._model_config_variant.model_config.get_config()]
        )

        model_configs.extend(
            [
                composing_config_variant.model_config.get_config()
                for composing_config_variant in self._composing_config_variants
            ]
        )

        return model_configs

    def is_legal_combination(self):
        """
        Returns true if the run_config is valid and should be run. Else false
        """
        legal = (
            self._check_for_client_vs_model_batch_size()
            and self._check_for_preferred_vs_model_batch_size()
        )

        return legal

    def is_ensemble_model(self) -> bool:
        """
        Returns true if the model config is an ensemble model
        """
        return self._model_config_variant.model_config.is_ensemble()

    def is_bls_model(self) -> bool:
        """
        Returns true if the model config is a BLS model
        """
        # If composing configs are present and it's not an ensemble it must be a BLS
        # Note: this will need to change if we allow ensembles to contain BLS models
        return (
            not self._model_config_variant.model_config.is_ensemble()
            and len(self._composing_config_variants) > 0
        )

    def get_composing_config_names(self) -> Optional[List[str]]:
        """
        Returns list of composing config names
        """
        return [
            composing_config_variant.variant_name
            for composing_config_variant in self._composing_config_variants
        ]

    def add_composing_model_config_variants(
        self, composing_model_config_variants: List[ModelConfigVariant]
    ) -> None:
        """
        Adds a list of composing model config variants
        """
        for composing_model_config_variant in composing_model_config_variants:
            self._composing_config_variants.append(composing_model_config_variant)

    @classmethod
    def from_dict(cls, model_run_config_dict):
        model_run_config = ModelRunConfig(None, None, None)
        model_run_config._model_name = model_run_config_dict["_model_name"]

        if "_model_config_variant" in model_run_config_dict:
            model_config = ModelConfig.from_dict(
                model_run_config_dict["_model_config_variant"]["model_config"]
            )
            variant_name = model_run_config_dict["_model_config_variant"][
                "variant_name"
            ]

            model_run_config._model_config_variant = ModelConfigVariant(
                model_config, variant_name
            )

        model_run_config._perf_config = PerfAnalyzerConfig.from_dict(
            model_run_config_dict["_perf_config"]
        )

        if "_composing_config_variants" in model_run_config_dict:
            model_run_config._composing_config_variants = [
                ModelConfigVariant(
                    ModelConfig.from_dict(
                        composing_config_variant_dict["model_config"]
                    ),
                    composing_config_variant_dict["variant_name"],
                )
                for composing_config_variant_dict in model_run_config_dict[
                    "_composing_config_variants"
                ]
            ]

        return model_run_config
