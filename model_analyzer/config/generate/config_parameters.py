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

from typing import Any, List, Optional, Tuple

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

from .config_parameter import ConfigParameter, ParameterCategory, ParameterType


class ConfigParameters:
    """
    Contains information about all configuration parameters the user wants to search
    """

    def __init__(self):
        self._parameters: Dict[str, ConfigParameter] = {}

    def add_parameter(
        self,
        name: str,
        ptype: ParameterType,
        category: ParameterCategory,
        min_range: Optional[int] = None,
        max_range: Optional[int] = None,
        enumerated_list: List[Any] = [],
    ):
        self._check_for_illegal_input(category, min_range, max_range, enumerated_list)

        self._parameters[name] = ConfigParameter(
            ptype, category, min_range, max_range, enumerated_list
        )

    def get_parameter(self, name: str) -> ConfigParameter:
        return self._parameters[name]

    def get_type(self, name: str) -> ParameterType:
        return self._parameters[name].ptype

    def get_category(self, name: str) -> ParameterCategory:
        return self._parameters[name].category

    def get_range(self, name: str) -> Tuple[int, int]:
        return (self._parameters[name].min_range, self._parameters[name].max_range)

    def get_list(self, name: str) -> List[Any]:
        return self._parameters[name].enumerated_list

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
                f"enumerated_list must be specified for a LIST"
            )
        elif min_range is not None:
            raise TritonModelAnalyzerException(
                f"min_range cannot be specified for a list"
            )
        elif max_range is not None:
            raise TritonModelAnalyzerException(
                f"max_range cannot be specified for a list"
            )
