# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.triton.model.model_config import ModelConfig


@dataclass
class ModelConfigVariant:
    """
    A dataclass that holds the ModelConfig as well as the variant name
    and cpu_only flag for the model
    """

    model_config: ModelConfig
    variant_name: str
    cpu_only: bool = False
