# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from .mock_base import MockBase
from unittest.mock import MagicMock, patch


class MockPSUtil(MockBase):

    def _fill_patchers(self):
        mock_process = MagicMock()
        mock_process().cpu_percent.return_value = 5
        self._patchers.append(
            patch('model_analyzer.perf_analyzer.perf_analyzer.psutil.Process',
                  mock_process))
