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
from unittest.mock import Mock, MagicMock, patch


class MockOSMethods(MockBase):
    """
    Mocks the methods for the os module
    """
    def __init__(self, mock_paths):
        path_attrs = {
            'join': MagicMock(return_value=""),
            'abspath': MagicMock(return_value=""),
            'isdir': MagicMock(return_value=True),
            'exists': MagicMock(return_value=True),
            'isfile': MagicMock(return_value=True)
        }
        os_attrs = {
            'path': Mock(**path_attrs),
            'mkdir': MagicMock(),
            'makedirs': MagicMock(),
            'getenv': MagicMock(),
            'listdir': MagicMock(return_value=[])
        }
        self._mock_paths = mock_paths
        self._patchers_os = {}
        self._os_mocks = {}
        for path in mock_paths:
            self._patchers_os[path] = patch(path, Mock(**os_attrs))
        super().__init__()

    def start(self):
        """
        start the patchers
        """

        for path in self._mock_paths:
            self._os_mocks[path] = self._patchers_os[path].start()

    def _fill_patchers(self):
        """
        Fills the patcher list for destruction
        """

        for patcher in self._patchers_os.values():
            self._patchers.append(patcher)
