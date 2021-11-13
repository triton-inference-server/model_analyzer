# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import patch
import sys

from .common import test_result_collector as trc

from model_analyzer.cli.cli import CLI


class TestCLI(trc.TestResultCollector):
    """
    Tests the methods of the CLI class
    """

    @patch('model_analyzer.cli.cli.ArgumentParser.print_help')
    def test_help_message_no_args(self, mock_print_help):
        """
        Tests that model-analyzer prints the help message when no arguments are
        given
        """

        sys.argv = ['/usr/local/bin/model-analyzer']

        cli = CLI()

        self.assertRaises(SystemExit, cli.parse)
        mock_print_help.assert_called()
