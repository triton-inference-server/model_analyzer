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

from collections import defaultdict
import argparse
import yaml
import sys
import os


class TestOutputValidator:
    """
    Functions that validate the output
    of the L0_results test
    """
    def __init__(self, config, test_name, export_path, model_names,
                 analyzer_log):
        self._config = config
        self._model_names = model_names.split(',')
        self._analyzer_log = analyzer_log
        self._export_path = export_path

        check_function = self.__getattribute__(f'check_{test_name}')

        if check_function():
            sys.exit(0)
        else:
            sys.exit(1)

    def check_num_checkpoints(self):
        """
        Open the checkpoints directory and 
        check that there is 5 checkpoints
        """

        checkpoint_files = os.listdir(
            os.path.join(self._export_path, 'checkpoints'))
        return len(checkpoint_files) == len(self._model_names)

    def check_loading_checkpoints(self):
        """
        Open the analyzer log and and make sure no perf
        analyzer runs took place
        """

        with open(self._analyzer_log, 'r') as f:
            log_contents = f.read()

        token = "Profiling model "
        return log_contents.find(token) == -1

    def check_interrupt_handling(self):
        """
        Open the checkpoints file and make sure there
        are only 3 checkpoints. Additionally
        check the analyzer log for a SIGINT.
        Also check that the 3rd model has
        been run once
        """

        checkpoint_files = os.listdir(
            os.path.join(self._export_path, 'checkpoints'))
        if len(checkpoint_files) != 3:
            return False

        with open(self._analyzer_log, 'r') as f:
            log_contents = f.read()

        # check for SIGINT
        token = "SIGINT"
        if log_contents.find(token) == -1:
            return False

        # check that 3rd model is profiled once
        token = f"Profiling model {self._model_names[2]}"
        token_idx = 0
        found_count = 0
        while True:
            token_idx = log_contents.find(token, token_idx + 1)
            if token_idx == -1:
                break
            found_count += 1

        return found_count == 1

    def check_continue_after_checkpoint(self):
        """
        Check that the 3rd model has been run once 
        and the remaining have been run twice each.
        """

        profiled_models = self._model_names[-3:]
        with open(self._analyzer_log, 'r') as f:
            log_contents = f.read()

        found_models_count = defaultdict(int)
        token_idx = 0
        while True:
            token_idx = log_contents.find('Profiling model ', token_idx + 1)
            if token_idx == -1:
                break
            end_of_model_name = log_contents.find('...', token_idx)
            model_name = log_contents[token_idx + len('Profiling model '
                                                      ):end_of_model_name]
            found_models_count[model_name.rsplit('_', 1)[0]] += 1

        for i in range(3):
            if found_models_count[profiled_models[i]] != 1:
                return False
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--config-file',
                        type=str,
                        required=True,
                        help='The path to the config yaml file.')
    parser.add_argument('-d',
                        '--export-path',
                        type=str,
                        required=True,
                        help='The export path for the model analyzer.')
    parser.add_argument('-m',
                        '--model-names',
                        type=str,
                        required=True,
                        help='The models being used for this test.')
    parser.add_argument('-l',
                        '--analyzer-log-file',
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
        config = yaml.load(f, Loader=yaml.FullLoader)

    TestOutputValidator(config, args.test_name, args.export_path,
                        args.model_names, args.analyzer_log_file)
