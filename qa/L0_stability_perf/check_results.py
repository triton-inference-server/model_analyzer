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
import sys
import glob
import os
from collections import defaultdict
import re


class TestOutputValidator:
    """
    Functions that validate the output
    of the test
    """
    def __init__(self, test_name, threshold):
        self._tolerance_percent = threshold

        check_function = self.__getattribute__(f'check_{test_name}')

        if check_function():
            sys.exit(0)
        else:
            sys.exit(1)

    def check_perf_stability(self):
        """
        Get the perf results from each model analyzer run
        as well as perf_analyzer run and compare them
        """

        log_paths = glob.glob(os.path.join(os.getcwd(), f"*.log"))

        # Split the logs up into log pairs model_name: [model_name.test.log, model_name.perf.log]
        # Note that the values may be ordered differently (perf log first then test log)
        log_pairs = defaultdict(list)
        for log_path in log_paths:
            model_name = os.path.relpath(log_path, os.getcwd()).split('.')[0]
            log_pairs[model_name].append(log_path)

        # compare the values within paired logs
        for model_name, log_pair in log_pairs.items():
            throughputs, latencies = [], []
            for log in sorted(log_pair):
                # Open log and match regex
                with open(log, 'r+') as f:
                    log_contents = f.read()

                throughput = re.search('Throughput: (\d+\.\d+|\d+)',
                                       log_contents)
                if throughput:
                    throughputs.append(float(throughput.group(1)))
                else:
                    print(f"\n***\n*** No throughput found in {log} \n***")
                    return False

                p99_latency = re.search('p99 latency: (\d+\.\d+|\d+)',
                                        log_contents)
                if p99_latency:
                    latencies.append(float(p99_latency.group(1)))
                else:
                    print(f"\n***\n*** No throughput found in {log} \n***")
                    return False

            # Once all throughputs and latencies are collected, compute diff percentage
            throughput_diff = 100 * (throughputs[0] -
                                     throughputs[1]) / throughputs[0]

            latency_diff = 100 * (latencies[1] - latencies[0]) / latencies[1]

            if throughput_diff > self._tolerance_percent and latency_diff > self._tolerance_percent:
                print(
                    f"\n***\n*** Model Analyzer throughput and latency differ "
                    f"greatly from perf analyzer standalone for model {model_name}. "
                    f"\n*** perf analyzer values: Throughput={throughputs[0]} infer/sec, p99 Latency={latencies[0]} usec. \n***"
                )
                return False
            elif throughput_diff > self._tolerance_percent:
                print(
                    f"\n***\n*** Model Analyzer throughput differs "
                    f"greatly from perf analyzer standalone for model {model_name}. \n***"
                    f"\n*** perf analyzer values: Throughput={throughputs[0]} infer/sec. \n***"
                )
                return False
            elif latency_diff > self._tolerance_percent:
                print(
                    f"\n***\n*** Model Analyzer latency differs "
                    f"greatly from perf analyzer standalone for model {model_name}. \n***"
                    f"\n*** perf analyzer values: p99 Latency={latencies[0]} usec. \n***"
                )
                return False
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--test-name',
                        type=str,
                        required=True,
                        help='The name of the test to be run.')
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1,
        help=
        'The allowed percentage difference of model analyzer metrics from perf_analyzer metrics. '
    )
    args = parser.parse_args()

    TestOutputValidator(args.test_name, args.tolerance)
