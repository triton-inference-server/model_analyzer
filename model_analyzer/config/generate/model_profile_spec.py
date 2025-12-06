#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Optional

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.objects.config_model_profile_spec import (
    ConfigModelProfileSpec,
)
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.triton.model.model_config import ModelConfig


class ModelProfileSpec(ConfigModelProfileSpec):
    """
    The profile configuration and default model config for a single model to be profiled
    """

    def __init__(
        self,
        spec: ConfigModelProfileSpec,
        config: ConfigCommandProfile,
        client: TritonClient,
        gpus: List[GPUDevice],
    ):
        # Determine cpu_only based on priority (before copying spec's __dict__):
        # 1) User-specified kind in instance_group (highest)
        # 2) cpu_only_composing_models config
        # 3) Inherited from spec (default False)
        explicit_kind = self._get_explicit_instance_kind(spec)
        if explicit_kind == "KIND_CPU":
            cpu_only = True
        elif explicit_kind == "KIND_GPU":
            cpu_only = False
        elif spec.model_name() in config.cpu_only_composing_models:
            cpu_only = True
        else:
            cpu_only = spec.cpu_only()

        super().__init__(spec.model_name(), cpu_only=cpu_only)

        # Copy remaining attributes from spec (excluding _cpu_only which we set above)
        spec_dict = deepcopy(spec.__dict__)
        spec_dict.pop("_cpu_only", None)
        self.__dict__.update(spec_dict)

        self._default_model_config = ModelConfig.create_model_config_dict(
            config, client, gpus, config.model_repository, spec.model_name()
        )

    @staticmethod
    def _get_explicit_instance_kind(spec: ConfigModelProfileSpec) -> Optional[str]:
        """
        Check if the spec has an explicit kind specified in instance_group.

        Returns the kind if explicitly specified, None otherwise.
        This allows users to specify KIND_CPU or KIND_GPU directly in
        model_config_parameters.instance_group instead of using the
        separate cpu_only_composing_models config option.

        The config parser may wrap values in lists for sweep support, so we need
        to handle structures like:
        - [[{'count': [1, 2, 4], 'kind': ['KIND_CPU']}]]  (double-wrapped, kind is list)
        - [{'count': [1, 2, 4], 'kind': 'KIND_CPU'}]      (single-wrapped, kind is string)
        """
        model_config_params = spec.model_config_parameters()
        if model_config_params is None:
            return None

        instance_group = model_config_params.get("instance_group")
        if instance_group is None or not isinstance(instance_group, list):
            return None

        # instance_group structure can be doubly wrapped due to config parsing:
        # [[ {'kind': ['KIND_GPU'], 'count': [1, 2, 4]} ]]
        # Unwrap the nested structure if needed
        if len(instance_group) > 0 and isinstance(instance_group[0], list):
            instance_group = instance_group[0]

        # instance_group is now a list of dicts, each potentially containing 'kind'
        for ig in instance_group:
            if isinstance(ig, dict):
                kind = ig.get("kind")
                # Handle case where kind is wrapped in a list by config parser
                # e.g., ['KIND_CPU'] instead of 'KIND_CPU'
                if isinstance(kind, list) and len(kind) > 0:
                    kind = kind[0]
                if kind in ("KIND_CPU", "KIND_GPU"):
                    return kind

        return None

    def get_default_config(self) -> dict:
        """Returns the default configuration for this model"""
        return deepcopy(self._default_model_config)

    def supports_batching(self) -> bool:
        """Returns True if this model supports batching. Else False"""
        if (
            "max_batch_size" not in self._default_model_config
            or self._default_model_config["max_batch_size"] == 0
        ):
            return False
        return True

    def supports_dynamic_batching(self) -> bool:
        """Returns True if this model supports dynamic batching. Else False"""
        supports_dynamic_batching = self.supports_batching()

        if "sequence_batching" in self._default_model_config:
            supports_dynamic_batching = False
        return supports_dynamic_batching

    def is_ensemble(self) -> bool:
        """Returns true if the model is an ensemble"""
        return "ensemble_scheduling" in self._default_model_config

    def is_load_specified(self) -> bool:
        """
        Returns true if the model's PA config has specified any of the
        inference load args (such as concurrency). Else returns false
        """
        load_args = PerfAnalyzerConfig.get_inference_load_args()
        pa_flags = self.perf_analyzer_flags()
        if pa_flags is None:
            return False
        return any(e in pa_flags for e in load_args)
