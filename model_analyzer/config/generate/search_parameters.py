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

from .search_parameter import ParameterCategory, ParameterUsage, SearchParameter


class SearchParameters:
    """
    Contains information about all configuration parameters the user wants to search
    """

    # These map to the run-config-search fields
    # See github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md
    exponential_rcs_parameters = ["batch_sizes", "concurrency"]
    linear_rcs_parameters = ["instance_group"]

    model_parameters = ["batch_sizes", "instance_group", "max_queue_delay_microseconds"]
    runtime_parameters = ["concurrency"]

    def __init__(
        self,
        config: ConfigCommandProfile = ConfigCommandProfile(),
        parameters: Dict[str, Any] = {},
        model_config_parameters: Dict[str, Any] = {},
    ):
        self._config = config
        self._parameters = parameters
        self._model_config_parameters = model_config_parameters
        self._search_parameters: Dict[str, SearchParameter] = {}

        self._populate_search_parameters()

    def get_parameter(self, name: str) -> Optional[SearchParameter]:
        try:
            parameter = self._search_parameters[name]
        except KeyError:
            return None

        return parameter

    def get_type(self, name: str) -> ParameterUsage:
        return self._search_parameters[name].usage

    def get_category(self, name: str) -> ParameterCategory:
        return self._search_parameters[name].category

    def get_range(self, name: str) -> Tuple[Optional[int], Optional[int]]:
        return (
            self._search_parameters[name].min_range,
            self._search_parameters[name].max_range,
        )

    def get_list(self, name: str) -> Optional[List[Any]]:
        return self._search_parameters[name].enumerated_list

    def _populate_search_parameters(self) -> None:
        if self._parameters:
            self._populate_parameters()
        else:
            # Always populate batch size if nothing is specified
            self._populate_batch_sizes()

        self._populate_model_config_parameters()

    def _populate_parameters(self) -> None:
        self._populate_batch_sizes()
        self._populate_concurrency()
        # TODO: Populate request rate - TMA-1903

    def _populate_model_config_parameters(self) -> None:
        self._populate_instance_group()
        self._populate_max_queue_delay_microseconds()

    def _populate_batch_sizes(self) -> None:
        if self._parameters and self._parameters["batch_sizes"]:
            self._populate_list_parameter(
                parameter_name="batch_sizes",
                parameter_list=self._parameters["batch_sizes"],
                parameter_category=ParameterCategory.INT_LIST,
            )
        else:
            self._populate_rcs_parameter(
                parameter_name="batch_sizes",
                rcs_parameter_min_value=self._config.run_config_search_min_model_batch_size,
                rcs_parameter_max_value=self._config.run_config_search_max_model_batch_size,
            )

    def _populate_concurrency(self) -> None:
        if self._parameters["concurrency"]:
            self._populate_list_parameter(
                parameter_name="concurrency",
                parameter_list=self._parameters["concurrency"],
                parameter_category=ParameterCategory.INT_LIST,
            )
        else:
            self._populate_rcs_parameter(
                parameter_name="concurrency",
                rcs_parameter_min_value=self._config.run_config_search_min_concurrency,
                rcs_parameter_max_value=self._config.run_config_search_max_concurrency,
            )

    def _populate_instance_group(self) -> None:
        # Example config format:
        #
        # model_config_parameters:
        #   instance_group:
        #     - kind: KIND_GPU
        #       count: [1, 2, 3, 4]

        # Need to populate instance_group based on RCS min/max values
        # even if no model config parameters are present
        if not self._model_config_parameters:
            self._populate_rcs_parameter(
                parameter_name="instance_group",
                rcs_parameter_min_value=self._config.run_config_search_min_instance_count,
                rcs_parameter_max_value=self._config.run_config_search_max_instance_count,
            )
        elif "instance_group" in self._model_config_parameters.keys():
            parameter_list = self._model_config_parameters["instance_group"][0][0][
                "count"
            ]

            self._populate_list_parameter(
                parameter_name="instance_group",
                parameter_list=parameter_list,
                parameter_category=ParameterCategory.INT_LIST,
            )
        else:
            self._populate_rcs_parameter(
                parameter_name="instance_group",
                rcs_parameter_min_value=self._config.run_config_search_min_instance_count,
                rcs_parameter_max_value=self._config.run_config_search_max_instance_count,
            )

    def _populate_max_queue_delay_microseconds(self) -> None:
        # Example format
        #
        # model_config_parameters:
        #  dynamic_batching:
        #    max_queue_delay_microseconds: [100, 200, 300]

        # There is no RCS field for max_queue_delay_microseconds
        if self._is_max_queue_delay_in_model_config_parameters():
            self._populate_list_parameter(
                parameter_name="max_queue_delay_microseconds",
                parameter_list=self._model_config_parameters["dynamic_batching"][0][
                    "max_queue_delay_microseconds"
                ],
                parameter_category=ParameterCategory.INT_LIST,
            )

    def _is_max_queue_delay_in_model_config_parameters(self) -> bool:
        if self._model_config_parameters:
            max_queue_delay_present = (
                "dynamic_batching" in self._model_config_parameters.keys()
                and (
                    "max_queue_delay_microseconds"
                    in self._model_config_parameters["dynamic_batching"][0]
                )
            )
        else:
            max_queue_delay_present = False

        return max_queue_delay_present

    def _populate_list_parameter(
        self,
        parameter_name: str,
        parameter_list: List[int | str],
        parameter_category: ParameterCategory,
    ) -> None:
        usage = self._determine_parameter_usage(parameter_name)

        self._add_search_parameter(
            name=parameter_name,
            usage=usage,
            category=parameter_category,
            enumerated_list=parameter_list,
        )

    def _populate_rcs_parameter(
        self,
        parameter_name: str,
        rcs_parameter_min_value: int,
        rcs_parameter_max_value: int,
    ) -> None:
        usage = self._determine_parameter_usage(parameter_name)
        category = self._determine_parameter_category(parameter_name)

        if category == ParameterCategory.EXPONENTIAL:
            min_range = int(log2(rcs_parameter_min_value))  # type: ignore
            max_range = int(log2(rcs_parameter_max_value))  # type: ignore
        else:
            min_range = rcs_parameter_min_value  # type: ignore
            max_range = rcs_parameter_max_value  # type: ignore

        self._add_search_parameter(
            name=parameter_name,
            usage=usage,
            category=category,
            min_range=min_range,
            max_range=max_range,
        )

    def _determine_parameter_category(self, name: str) -> ParameterCategory:
        if name in SearchParameters.exponential_rcs_parameters:
            category = ParameterCategory.EXPONENTIAL
        elif name in SearchParameters.linear_rcs_parameters:
            category = ParameterCategory.INTEGER
        else:
            TritonModelAnalyzerException(f"ParameterCategory not found for {name}")

        return category

    def _determine_parameter_usage(self, name: str) -> ParameterUsage:
        if name in SearchParameters.model_parameters:
            usage = ParameterUsage.MODEL
        elif name in SearchParameters.runtime_parameters:
            usage = ParameterUsage.RUNTIME
        else:
            TritonModelAnalyzerException(f"ParameterUsage not found for {name}")

        return usage

    def _add_search_parameter(
        self,
        name: str,
        usage: ParameterUsage,
        category: ParameterCategory,
        min_range: Optional[int] = None,
        max_range: Optional[int] = None,
        enumerated_list: List[Any] = [],
    ) -> None:
        self._check_for_illegal_input(category, min_range, max_range, enumerated_list)

        self._search_parameters[name] = SearchParameter(
            usage=usage,
            category=category,
            enumerated_list=enumerated_list,
            min_range=min_range,
            max_range=max_range,
        )

    def _check_for_illegal_input(
        self,
        category: ParameterCategory,
        min_range: Optional[int],
        max_range: Optional[int],
        enumerated_list: List[Any],
    ) -> None:
        if (
            category is ParameterCategory.INT_LIST
            or category is ParameterCategory.STR_LIST
        ):
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