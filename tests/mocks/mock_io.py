# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock, MagicMock, patch, mock_open


class MockIOMethods:
    """
    A class that mocks filesystem
    operations open and write
    """

    def __init__(self):
        self.patcher_open = patch('model_analyzer.output.file_writer.open',
                                  mock_open())
        self.patcher_print = patch('model_analyzer.output.file_writer.print',
                                   MagicMock())
        self.open_mock = self.patcher_open.start()
        self.print_mock = self.patcher_print.start()

    def stop(self):
        """
        Stops the mock of io functions
        in file_writer
        """

        self.patcher_open.stop()
        self.patcher_print.stop()

    def raise_exception_on_open(self):
        """
        Raises an OSError when open is called
        """
        self.open_mock.side_effect = OSError

    def raise_exception_on_write(self):
        """
        Raises an OSError when write is called
        """
        self.open_mock.return_value.write.side_effect = OSError

    def assert_open_called_with_args(self, filename):
        """
        Asserts that file open was called
        with given arguments 
        """

        self.open_mock.assert_called_with(out)

    def assert_write_called_with_args(self, out):
        """
        Asserts that file write was called
        with given arguments 
        """

        self.open_mock.return_value.write.assert_called_with(out)

    def assert_print_called_with_args(self, out):
        """
        Asserts that print was called
        with given arguments
        """

        self.print_mock.assert_called_with(out, end='')

    def reset(self):
        self.open_mock.side_effect = None
        self.open_mock.return_value.write.side_effect = None
