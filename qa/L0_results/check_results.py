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

import argparse
import yaml
import os
import sys


class TestOutputValidator:
    """
    Functions that validate the output of
    the test
    """
    def __init__(self, config, test_name, export_path):
        self._config = config
        self._export_path = export_path
        check_function = self.__getattribute__(f'check_{test_name}')

        if check_function():
            sys.exit(0)
        else:
            sys.exit(1)

    def check_summaries(self):
        """
        Checks if the result summary pdfs exists
        in the required location

        Returns
        -------
        True if all exist, False otherwise
        """

        for model in config['analysis_models']:
            if not os.path.exists(
                    os.path.join(self._export_path, 'reports', 'summaries',
                                 model, 'result_summary.pdf')):
                print(f"\n***\n*** Summary not found for {model}.\n***")
                return False

        # First check for the best models report
        analysis_models = set(config['analysis_models'])
        report_dirs = set(
            os.listdir(os.path.join(self._export_path, 'reports',
                                    'summaries')))
        if len(report_dirs - analysis_models) != 1:
            print("\n***\n*** Top models summary not found.\n***")
            return False

        # Should be only 1 element in intersection
        for dir in (report_dirs - analysis_models):
            if not os.path.exists(
                    os.path.join(self._export_path, 'reports', 'summaries',
                                 dir, 'result_summary.pdf')):
                print("\n***\n*** Top models summary not found.\n***")
                return False

        return True

    def check_detailed_reports(self):
        """
        Checks if the detailed report pdfs exists
        in the required location
        """

        for model_config in config['report_model_configs']:
            if not os.path.exists(
                    os.path.join(self._export_path, 'reports', 'detailed',
                                 model_config, 'detailed_report.pdf')):
                print(
                    f"\n***\n*** Detailed report not found for {model_config}.\n***"
                )
                return False
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--config-file',
                        type=str,
                        required=True,
                        help='The config file for this test')
    parser.add_argument('-d',
                        '--export-path',
                        type=str,
                        required=True,
                        help='The full path to the export directory')
    parser.add_argument('-t',
                        '--test-name',
                        type=str,
                        required=True,
                        help='The name of the test to be run.')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    TestOutputValidator(config, args.test_name, args.export_path)
