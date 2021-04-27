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


class MockPickleMethods(MockBase):
    """
    Mocks the methods for the os module
    """
    def __init__(self):
        pickle_attrs = {'load': MagicMock(), 'dump': MagicMock()}
        self.patcher_pickle = patch(
            'model_analyzer.state.analyzer_state_manager.pickle',
            Mock(**pickle_attrs))
        super().__init__()

    def start(self):
        """
        start the patchers
        """

        self.pickle_mock = self.patcher_pickle.start()

    def _fill_patchers(self):
        """
        Fills the patcher list for destruction
        """

        self._patchers.append(self.patcher_pickle)

    def set_pickle_load_return_value(self, value):
        """
        Sets the return value for pickle load
        """

        self.pickle_mock.load.return_value = value

    def set_pickle_load_side_effect(self, effect):
        """
        Sets a side effet
        """

        self.pickle_mock.load.side_effect = effect
