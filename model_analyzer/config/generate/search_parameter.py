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

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional


class ParameterUsage(Enum):
    MODEL = auto()
    RUNTIME = auto()
    BUILD = auto()


class ParameterCategory(Enum):
    INTEGER = auto()
    EXPONENTIAL = auto()
    STR_LIST = auto()
    INT_LIST = auto()


@dataclass
class SearchParameter:
    """
    A dataclass that holds information about a configuration's search parameter
    """

    usage: ParameterUsage
    category: ParameterCategory

    # This is only applicable to the LIST categories
    enumerated_list: Optional[List[Any]] = None

    # These are only applicable to INTEGER and EXPONENTIAL categories
    min_range: Optional[int] = None
    max_range: Optional[int] = None
