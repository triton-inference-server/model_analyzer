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


def add_arguments(parser):
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


def check_summary(config, export_path):
    """
    Checks if the result summary pdfs exists
    in the required location

    Returns
    -------
    True if all exist, False otherwise
    """

    for model in config['model_names']:
        if not os.path.exists(
                os.path.join(export_path, 'reports', model,
                             'result_summary.pdf')):
            print(f"\n***\n*** Summary not found for {model}.\n***")
            return False
    return True


def check_best_models(config, export_path):
    """
    Checks if the required number of best model 
    folders exist in the required location

    Returns
    -------
    True if correct number exist, False otherwise
    """

    # First check for the best models report
    model_names = set(config['model_names'].keys())
    report_dirs = set(os.listdir(os.path.join(export_path, 'reports')))
    if len(report_dirs - model_names) != 1:
        print("\n***\n*** Top models summary not found.\n***")
        return False

    # Should be only 1 element in intersection
    for dir in (report_dirs - model_names):
        if not os.path.exists(
                os.path.join(export_path, 'reports', dir,
                             'result_summary.pdf')):
            print("\n***\n*** Top models summary not found.\n***")
            return False

    top_model_dir = os.path.join(export_path, 'best_models')
    if not os.path.exists(top_model_dir):
        return True
    num_best_models = len(os.listdir(top_model_dir))
    if config['num_top_model_configs'] == num_best_models:
        return True
    else:
        print(
            f"\n***\n*** Expected {config['num_top_model_configs']} best models. Found {num_best_models}.\n***"
        )
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    passed_test = True
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if 'summarize' in config and config['summarize']:
        passed_test &= check_summary(config, args.export_path)

    if 'num_top_model_configs' in config and config[
            'num_top_model_configs'] != 0:
        passed_test &= check_best_models(config, args.export_path)

    if passed_test:
        sys.exit(0)
    else:
        sys.exit(1)
