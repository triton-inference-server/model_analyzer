# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import abc
from typing import List, Optional, Generator, Any
from model_analyzer.result.run_config_measurement import RunConfigMeasurement


class ConfigGeneratorInterface(abc.ABC):
    """
    An interface class for config generators
    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> bool:
        return (hasattr(subclass, '__init__') and \
                callable(subclass.__init__) and \
                hasattr(subclass, 'get_configs') and \
                callable(subclass.get_configs) and \
                hasattr(subclass, 'set_last_results') and \
                callable(subclass.set_last_results) or \
                NotImplemented)

    @abc.abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_configs(self) -> Generator[Any, None, None]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_last_results(
            self, measurements: List[Optional[RunConfigMeasurement]]) -> None:
        raise NotImplementedError
