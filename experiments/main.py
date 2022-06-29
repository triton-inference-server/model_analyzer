# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#####################
#
# Example usage:
#
# python3 main.py --model-name resnet50_libtorch --generator BruteRunConfigGenerator --run-config-search-max-model-batch-size 2
# python3 main.py --model-name resnet50_libtorch --generator QuickRunConfigGenerator --magnitude 5
#####################

from evaluate_config_generator import EvaluateConfigGenerator
from model_analyzer.constants import LOGGER_NAME
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v',
                    '--verbose',
                    required=False,
                    action='store_true',
                    help='Enable MA logging')
parser.add_argument("--model-name",
                    type=str,
                    required=True,
                    help="The model name")
parser.add_argument("--data-path",
                    type=str,
                    required=False,
                    default="./data",
                    help="The path to the checkpoint results files")
parser.add_argument("--output-path",
                    type=str,
                    required=False,
                    default="./output",
                    help="The path to the output csv files")
parser.add_argument("--generator",
                    type=str,
                    required=True,
                    help="The name of the config generator to evaluate")
args, other_args = parser.parse_known_args()

if args.verbose:
    import logging
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level=logging.DEBUG)
    logging.basicConfig(format="[Model Analyzer] %(message)s")

ecg = EvaluateConfigGenerator(args.model_name, args.data_path, args.output_path,
                              other_args)
ecg.execute_generator(args.generator)
ecg.print_results()
ecg.store_results()
