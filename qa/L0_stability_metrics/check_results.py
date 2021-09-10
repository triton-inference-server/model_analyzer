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

    def __init__(self, config, test_name, results_path, tolerance):
        self._config = config
        self._models = list(config['profile_models'])
        self._result_path = results_path
        self._tolerance = tolerance

        check_function = self.__getattribute__(f'check_{test_name}')

        if check_function():
            sys.exit(0)
        else:
            sys.exit(1)

    def check_metrics_stability(self):
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

        # Now in the first csv, get the metrics
        metric_values = {}
        for csv in csv_contents:
            csv_lines = csv.split('\n')
            for line in csv_lines[1:-2]:
                model, _, _, gpu_memory, gpu_utilization = line.split(',')
                if model in metric_values:
                    metric_values[model]['gpu_memory'].append(float(gpu_memory))
                    metric_values[model]['gpu_utilization'].append(
                        float(gpu_utilization))
                else:
                    metric_values[model] = {
                        'gpu_memory': [float(gpu_memory)],
                        'gpu_utilization': [float(gpu_utilization)]
                    }

        # Compare metrics
        for model in metric_values:
            for metric, values in metric_values[model].items():
                start_value = values[0]
                for value in values[1:]:
                    deviation_percent = abs(
                        (value - start_value) / start_value) * 100
                    if deviation_percent > self._tolerance:
                        print(
                            f"\n***"
                            f"\n***  For model {model}, value for metric {metric}"
                            "\n***  is unstable.\n***\n"
                            f"\n***\n***  Expected: {start_value} +/- {self._tolerance*start_value/100}."
                            f"\n***  Found: {values[1:]}.\n***")
                        return False
        return True


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
    parser.add_argument(
        '--tolerance',
        type=int,
        default=10,
        help='The percent tolerance allowed for the metrics to vary.')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    TestOutputValidator(config, args.test_name, args.inference_results_path,
                        args.tolerance)
