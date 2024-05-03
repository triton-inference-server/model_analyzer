#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from math import log2
from typing import Any, Dict, List, Optional, Tuple

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

from .config_parameter import ConfigParameter, ParameterCategory, ParameterType


class ConfigParameters:
    """
    Contains information about all configuration parameters the user wants to search
    """

    exponential_parameters = ["batch_sizes", "concurrency"]
    linear_parameters = ["instance_group"]

    model_parameters = ["batch_sizes", "instance_group", "max_queue_delay_microseconds"]
    runtime_parameters = ["concurrency"]

    def __init__(
        self,
        config: Optional[ConfigCommandProfile] = None,
        parameters: Dict[str, Any] = {},
        model_config_parameters: Dict[str, Any] = {},
    ):
        self._parameters: Dict[str, ConfigParameter] = {}

        if config:
            self.populate_search_parameters(config, parameters, model_config_parameters)

    def get_parameter(self, name: str) -> ConfigParameter:
        return self._parameters[name]

    def get_type(self, name: str) -> ParameterType:
        return self._parameters[name].ptype

    def get_category(self, name: str) -> ParameterCategory:
        return self._parameters[name].category

    def get_range(self, name: str) -> Tuple[Optional[int], Optional[int]]:
        return (self._parameters[name].min_range, self._parameters[name].max_range)

    def get_list(self, name: str) -> Optional[List[Any]]:
        return self._parameters[name].enumerated_list

    def populate_search_parameters(
        self,
        config: ConfigCommandProfile,
        parameters: Dict[str, Any],
        model_config_parameters: Dict[str, Any],
    ) -> None:
        self._populate_parameters(config, parameters)
        self._populate_model_config_parameters(config, model_config_parameters)

    def _populate_parameters(
        self,
        config: ConfigCommandProfile,
        parameters: Dict[str, Any],
    ) -> None:
        self._populate_parameter(
            parameter_name="batch_sizes",
            rcs_min_value=config.run_config_search_min_model_batch_size,
            rcs_max_value=config.run_config_search_max_model_batch_size,
            parameter_list=parameters["batch_sizes"],
        )
        # TODO: Figure out how to use request rate
        self._populate_parameter(
            parameter_name="concurrency",
            rcs_min_value=config.run_config_search_min_concurrency,
            rcs_max_value=config.run_config_search_max_concurrency,
            parameter_list=parameters["concurrency"],
        )

    def _populate_parameter(
        self,
        parameter_name: str,
        rcs_min_value: Optional[int] = None,
        rcs_max_value: Optional[int] = None,
        parameter_list: Optional[List[int]] = None,
    ) -> None:
        ptype = self._determine_parameter_type(parameter_name)

        if parameter_list:
            self._add_parameter(
                name=parameter_name,
                ptype=ptype,
                category=ParameterCategory.LIST,
                enumerated_list=parameter_list,
            )
        else:
            category = self._determine_parameter_category(parameter_name)

            if category == ParameterCategory.EXPONENTIAL:
                min_range = int(log2(rcs_min_value))  # type: ignore
                max_range = int(log2(rcs_max_value))  # type: ignore
            else:
                min_range = rcs_min_value  # type: ignore
                max_range = rcs_max_value  # type: ignore

            self._add_parameter(
                name=parameter_name,
                ptype=ptype,
                category=category,
                min_range=min_range,
                max_range=max_range,
            )

    def _determine_parameter_category(self, name: str) -> ParameterCategory:
        if name in ConfigParameters.exponential_parameters:
            category = ParameterCategory.EXPONENTIAL
        elif name in ConfigParameters.linear_parameters:
            category = ParameterCategory.INTEGER
        else:
            TritonModelAnalyzerException(f"ParameterCategory not found for {name}")

        return category

    def _determine_parameter_type(self, name: str) -> ParameterType:
        if name in ConfigParameters.model_parameters:
            ptype = ParameterType.MODEL
        elif name in ConfigParameters.runtime_parameters:
            ptype = ParameterType.RUNTIME
        else:
            TritonModelAnalyzerException(f"ParameterType not found for {name}")

        return ptype

    def _populate_model_config_parameters(
        self, config: ConfigCommandProfile, model_config_parameters: Dict[str, Any]
    ) -> None:
        # Need to populate instance_group based on RCS min/max values
        # even if no model config parameters are present
        if not model_config_parameters:
            self._populate_parameter(
                parameter_name="instance_group",
                rcs_min_value=config.run_config_search_min_instance_count,
                rcs_max_value=config.run_config_search_max_instance_count,
            )
            return

        if "instance_group" in model_config_parameters.keys():
            parameter_list = model_config_parameters["instance_group"][0][0]["count"]
        else:
            parameter_list = None

        self._populate_parameter(
            parameter_name="instance_group",
            rcs_min_value=config.run_config_search_min_instance_count,
            rcs_max_value=config.run_config_search_max_instance_count,
            parameter_list=parameter_list,
        )

        if "dynamic_batching" in model_config_parameters.keys():
            if (
                "max_queue_delay_microseconds"
                in model_config_parameters["dynamic_batching"][0]
            ):
                self._populate_parameter(
                    parameter_name="max_queue_delay_microseconds",
                    parameter_list=model_config_parameters["dynamic_batching"][0][
                        "max_queue_delay_microseconds"
                    ],
                )

    def _add_parameter(
        self,
        name: str,
        ptype: ParameterType,
        category: ParameterCategory,
        min_range: Optional[int] = None,
        max_range: Optional[int] = None,
        enumerated_list: List[Any] = [],
    ) -> None:
        self._check_for_illegal_input(category, min_range, max_range, enumerated_list)

        self._parameters[name] = ConfigParameter(
            ptype, category, min_range, max_range, enumerated_list
        )

    def _check_for_illegal_input(
        self,
        category: ParameterCategory,
        min_range: Optional[int],
        max_range: Optional[int],
        enumerated_list: List[Any],
    ) -> None:
        if category is ParameterCategory.LIST:
            self._check_for_illegal_list_input(min_range, max_range, enumerated_list)
        else:
            if min_range is None or max_range is None:
                raise TritonModelAnalyzerException(
                    f"Both min_range and max_range must be specified"
                )

            if min_range and max_range:
                if min_range > max_range:
                    raise TritonModelAnalyzerException(
                        f"min_range cannot be larger than max_range"
                    )

    def _check_for_illegal_list_input(
        self,
        min_range: Optional[int],
        max_range: Optional[int],
        enumerated_list: List[Any],
    ) -> None:
        if not enumerated_list:
            raise TritonModelAnalyzerException(
                f"enumerated_list must be specified for a ParameterCategory.LIST"
            )
        elif min_range is not None:
            raise TritonModelAnalyzerException(
                f"min_range cannot be specified for a ParameterCategory.LIST"
            )
        elif max_range is not None:
            raise TritonModelAnalyzerException(
                f"max_range cannot be specified for a ParameterCategory.LIST"
            )
