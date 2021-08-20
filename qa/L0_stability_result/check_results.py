# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import sys
import yaml
import os
import glob


class TestOutputValidator:
    """
    Functions that validate the output
    of the test
    """

    def __init__(self, config, test_name, results_path):
        self._config = config
        self._models = list(config['analysis_models'].keys())
        self._result_path = results_path

        check_function = self.__getattribute__(f'check_{test_name}')

        if check_function():
            sys.exit(0)
        else:
            sys.exit(1)

    def check_results_stability(self):
        """
        Makes sure that the same configuration
        appears as the best across iterations for
        each model
        """

        # There should be 4 csv files the results path
        pathname = os.path.join(self._result_path, 'result_*.csv')
        csv_contents = []
        for filename in glob.glob(pathname):
            with open(filename, 'r+') as f:
                csv_contents.append(f.read())

        # Now in the first csv, get the best rows for each model
        best_rows = []
        best_row_end = 0
        for model in self._models:
            best_row_start = csv_contents[0].find(model, best_row_end)
            best_row_end = csv_contents[0].find('\n', best_row_start + 1)
            best_rows.append(csv_contents[0][best_row_start:best_row_end])
        for csv in csv_contents[1:]:
            found_rows = []
            best_row_end = 0
            for model in self._models:
                best_row_start = csv.find(model)
                best_row_end = csv.find('\n', best_row_start + 1)
                found_rows.append(csv[best_row_start:best_row_end])

            # Compare the rows
            for i in range(len(best_rows)):
                if best_rows[i] != found_rows[i]:
                    self._print_diff(best_rows[i], found_rows[i], csv,
                                     self._models[i])
                    return False
        return True

    def _print_diff(self, best_row, found_row, csv, model):
        """
        Create and display a diff table
        """
        header_row = [''] + csv[:csv.find('\n')].split(',')
        expected_row = ["Expected:"] + best_row.split(',')
        found_row = ["Found:"] + found_row.split(',')

        # Pad cells
        for i in range(len(header_row)):
            cell_width = max(
                max(len(header_row[i]), len(expected_row[i]),
                    len(found_row[i])), 10)
            header_row[i] += max(cell_width - len(header_row[i]), 0) * " "
            expected_row[i] += max(cell_width - len(expected_row[i]), 0) * " "
            found_row[i] += max(cell_width - len(found_row[i]), 0) * " "
        header_diff, expected_diff, found_diff = [], [], []
        for i in range(len(expected_row)):
            if expected_row[i] != found_row[i]:
                header_diff.append(header_row[i])
                expected_diff.append(expected_row[i])
                found_diff.append(found_row[i])

        # Print message and table
        print(f"\n***"
              f"\n***  For model {model}, expected optimal"
              " config and found optimal config differ. "
              "\n***  Refer to the table below for details."
              "\n***")

        print(
            f"\n***  {'  '.join(header_diff)}\n***  {'  '.join(expected_diff)}\n***  {'  '.join(found_diff)}\n***"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--config-file',
                        type=str,
                        required=True,
                        help='The path to the config yaml file.')
    parser.add_argument('-r',
                        '--inference-results-path',
                        type=str,
                        required=True,
                        help='The full path to the analyzer log.')
    parser.add_argument('-t',
                        '--test-name',
                        type=str,
                        required=True,
                        help='The name of the test to be run.')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    TestOutputValidator(config, args.test_name, args.inference_results_path)
