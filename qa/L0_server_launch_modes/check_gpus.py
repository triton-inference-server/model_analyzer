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
import os


def check_gpus(analyzer_log, gpus, check_visible):
    """
    Creates the set of GPUs used by the analyzer
    and compares it to the expected_set
    """

    with open(analyzer_log, 'r') as f:
        log_contents = f.read()

    token = "Using GPU(s) with UUID(s) = {"
    gpus_start = log_contents.rfind(token)
    if gpus_start == -1:
        print(
            f"\n***\n*** Could not find GPUs used in the analyzer log {analyzer_log}.\n***"
        )
        sys.exit(1)

    gpus_start += len(token)
    gpus_end = log_contents.find('}', gpus_start + 1)
    analyzer_gpus = log_contents[gpus_start:gpus_end].strip().split(',')
    gpu_list = gpus.split(',')
    if check_visible:
        visible_indices = list(
            map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        expected_gpus = []
        for i in visible_indices:
            expected_gpus.append(gpu_list[i])
    else:
        expected_gpus = gpu_list

    if set(analyzer_gpus) == set(expected_gpus):
        sys.exit(0)
    else:
        print("\n***\n*** Model Analyzer is not using the correct GPUs.\n***")
        print(f"\n***\n*** Analyzer GPUS: {analyzer_gpus}.\n***")
        print(f"\n***\n*** Expected GPUS: {expected_gpus}.\n***")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyzer-log',
                        type=str,
                        required=True,
                        help="path to model analyzer log")
    parser.add_argument('--gpus',
                        type=str,
                        required=True,
                        help="Comma separated string with the expected GPUs")
    parser.add_argument('--check-visible',
                        action='store_true',
                        help='If expecting all visible GPUs')

    args = parser.parse_args()
    check_gpus(args.analyzer_log, args.gpus, args.check_visible)
