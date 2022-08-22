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

import os
import json
import subprocess
import re
from statistics import mean, median, mode

PATH = "/mnt/nvdl/datasets/inferenceserver/mnaas-checkpoints"


def find_all_checkpoints(path):
    """
    Return a list of all checkpoints at any depth below input path
    """
    all_checkpoints = []
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".ckpt"):
                all_checkpoints.append(f"{dirpath}/{filename}")
    return all_checkpoints


def get_models_in_checkpoint(ckpt):
    """
    Given a path to a checkpoint file, return a list of all model names
    inside of that checkpoint
    """
    if not os.path.exists(ckpt):
        print(f"File {ckpt} not found")
        exit(1)

    with open(ckpt, 'r') as f:
        contents = json.load(f)
        models = list(contents["ResultManager.results"]["_results"].keys())
    return models


def cleanup_checkpoints(path):
    """ 
    Given a path, recursively search all folders for all folders
    with one or more checkpoints in it. In each of those folders,
    rename the highest numbered checkpoint to be 0.ckpt and delete
    all others
    """
    for dirpath, _, filenames in os.walk(path):
        cleanup = False
        for filename in filenames:
            if filename.endswith(".ckpt"):
                cleanup = True
                break
        if cleanup:
            filenames.sort()
            os.rename(f"{dirpath}/{filenames[-1]}", f"{dirpath}/zzz")
            for filename in filenames[:-1]:
                os.remove(f"{dirpath}/{filename}")
            os.rename(f"{dirpath}/zzz", f"{dirpath}/0.ckpt")


def parse_experiment_output(output):
    """
    Returns a dict of key values from an experiment output
    """
    ret = {}

    output_str = output.decode('utf-8')
    lines = output_str.split("\n")
    for line in lines:
        if re.search(':', line):
            pair = line.split(":", 1)
            if pair[0] != "WARNING":
                ret[pair[0]] = pair[1].lstrip()

    return ret


def run_all_models():
    percentiles = []
    below_cutoff = []
    num_measurements = []
    too_many_measurements = []
    too_few_measurements = []

    PERCENTILE_CUTOFF = 0.70
    NUM_MEASUREMENTS_MAX = 15
    NUM_MEASUREMENTS_MIN = 5
    ckpts = find_all_checkpoints(PATH)
    total_models = 0

    # FIXME
    #ckpts = ckpts[80:82]
    for i, ckpt in enumerate(ckpts):

        # FIXME -- infinite loop
        if re.search("bert-base-cased-pyt", ckpt):
            continue

        # FIXME -- they don't use max_batch
        if re.search("ncf", ckpt):
            continue

        print(f"{i+1} out of {len(ckpts)} checkpoints")
        print(ckpt)

        is_linear_inst = False
        with open(ckpt) as f:
            contents = f.read()
            if re.search('instanceGroup": \[{"count": 3,', contents):
                is_linear_inst = True
        for model in get_models_in_checkpoint(ckpt):
            total_models += 1
            cmd = f"python3 /home/tgerdes/Code/model_analyzer/experiments/main.py --model-name {model} --generator QuickRunConfigGenerator --data-path {ckpt}"

            if not is_linear_inst:
                cmd = f"{cmd} --exponential-inst-count"
            cmd_list = cmd.split(' ')
            result = subprocess.run(cmd_list, stdout=subprocess.PIPE)
            result_data = parse_experiment_output(result.stdout)

            if 'Percentile' not in result_data:
                print(f"{model} failed! Skipping")
                continue
            if 'Generator num configs' not in result_data:
                print(f"{model} failed! Skipping")
                continue

            percentile = float(result_data['Percentile'])
            num_measurement = int(result_data['Generator num configs'])

            print(
                f"  {model}: Percentile = {percentile}, measurements = {num_measurement}"
            )
            percentiles.append(percentile)
            num_measurements.append(num_measurement)

            cmd = cmd.replace("main.py", "main.py -v")
            if percentile < PERCENTILE_CUTOFF:
                below_cutoff.append(f"{percentile}: {cmd}")
            if num_measurement > NUM_MEASUREMENTS_MAX:
                too_many_measurements.append(f"{num_measurement}: {cmd}")
            if num_measurement < NUM_MEASUREMENTS_MIN:
                too_few_measurements.append(f"{num_measurement}: {cmd}")

    avg_percentile = mean(percentiles)
    median_percentile = median(percentiles)
    mode_percentile = mode(percentiles)

    print()
    print(f"Average percentile = {avg_percentile:0.2f}")
    print(f"Median percentile = {median_percentile:0.2f}")
    print(f"Mode percentile = {mode_percentile:0.2f}")

    avg_measurements = mean(num_measurements)
    median_measurements = median(num_measurements)
    mode_measurements = mode(num_measurements)

    print()
    print(f"Average measurements = {avg_measurements:0.2f}")
    print(f"Median measurements = {median_measurements:0.2f}")
    print(f"Mode measurements = {mode_measurements:0.2f}")

    print(
        f"{len(below_cutoff)} out of {total_models} are below the cutoff of {100*PERCENTILE_CUTOFF}%:"
    )
    for x in sorted(below_cutoff):
        print(f"  {x}")

    print()
    print()
    print(
        f"{len(too_many_measurements)} out of {total_models} are taking more than {NUM_MEASUREMENTS_MAX} measurements:"
    )
    for x in sorted(too_many_measurements, reverse=True):
        print(f"  {x}")

    print()
    print()
    print(
        f"{len(too_few_measurements)} out of {total_models} are taking less than {NUM_MEASUREMENTS_MIN} measurements:"
    )
    for x in sorted(too_few_measurements):
        print(f"  {x}")


run_all_models()
