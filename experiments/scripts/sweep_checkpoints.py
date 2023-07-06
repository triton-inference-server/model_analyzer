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

FEW_CHECKPOINTS = False  # If true, only run 2 of the checkpoints instead of all
ONE_MODEL_EACH = True  # If true, only run 1 model from each checkpoint
THROUGHPUT_ONLY = False  # If true, only run maximize_throughput


class SweepResult:
    """ Holds result for a single experiment """

    def __init__(self, result_data, cmd, model) -> None:
        self.percentile = float(result_data['Percentile'])
        self.actual_measurements = int(
            result_data['Generator num measurements'])
        self.missing_measurements = int(
            result_data['Generator missing num measurements'])
        self.generator_best_latency = float(
            result_data['Generator best latency'])
        self.generator_best_throughput = float(
            result_data['Generator best throughput'])
        self.num_measurements = self.actual_measurements + self.missing_measurements
        self.cmd = cmd
        self.model = model


class SweepCheckpoints:

    def __init__(self) -> None:
        self.results = {
            "normal": {},
            "latency_budget": {},
            "min_throughput": {}
        }

    def run(self):
        self._run_all_models("normal")
        if not THROUGHPUT_ONLY:
            self._run_all_models("latency_budget")
            self._run_all_models("min_throughput")

        self._analyze("normal")
        if not THROUGHPUT_ONLY:
            self._analyze("latency_budget")
            self._analyze("min_throughput")

    def _find_all_checkpoints(self, path):
        """
        Return a list of all checkpoints at any depth below input path
        """
        all_checkpoints = []
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".ckpt"):
                    all_checkpoints.append(f"{dirpath}/{filename}")
        return all_checkpoints

    def _get_models_in_checkpoint(self, ckpt):
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

    def _parse_experiment_output(self, output):
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

    def _run_all_models(self, key):
        ckpts = self._find_all_checkpoints(PATH)
        total_models = 0

        if FEW_CHECKPOINTS:
            ckpts = ckpts[1:3]

        for i, ckpt in enumerate(ckpts):

            print(f"{i+1} out of {len(ckpts)} checkpoints")
            print(ckpt)

            with open(ckpt) as f:
                contents = f.read()
                is_linear_inst = self._is_linear_inst(contents)
                min_mbs_index = self._get_min_mbs_index(contents)

            models = self._get_models_in_checkpoint(ckpt)
            if ONE_MODEL_EACH:
                models = models[:1]

            for model in models:
                total_models += 1

                result = self._run_experiment(model, ckpt, min_mbs_index,
                                              is_linear_inst, key)
                if result:
                    self.results[key][ckpt + model] = result

    def _is_linear_inst(self, contents) -> bool:
        ret = False
        if re.search('instanceGroup": \[{"count": 3,', contents):
            ret = True
        return ret

    def _get_min_mbs_index(self, contents) -> bool:
        # If max batch size of 2 is not in the database, then find the index
        # where max batch sizes start (other than 1, which is part of default)
        min_mbs_index = 0
        if not re.search('maxBatchSize": 2,', contents):
            found = False
            min_mbs_index = 1
            while not found:
                min_mbs_index += 1
                min_mbs_search_val = 2**min_mbs_index
                search_str = f'maxBatchSize": {min_mbs_search_val},'
                found = re.search(search_str, contents)
        return min_mbs_index

    def _run_experiment(self, model, ckpt, min_mbs_index, is_linear_inst, key):
        cmd = f"python3 /home/tgerdes/Code/model_analyzer/experiments/main.py --model-name {model} --run-config-search-mode quick --data-path {ckpt} --min-mbs-index {min_mbs_index}"

        if not is_linear_inst:
            cmd = f"{cmd} --exponential-inst-count"

        if key != "normal" and ckpt + model not in self.results["normal"]:
            return None

        if key == "latency_budget":
            old_results = self.results["normal"][ckpt + model]
            lat_best = old_results.generator_best_latency
            lat_budget = int(lat_best * 0.9)
            cmd = f"{cmd} --latency-budget={lat_budget}"
        elif key == "min_throughput":
            old_results = self.results["normal"][ckpt + model]
            tput_best = old_results.generator_best_throughput
            min_tput = int(tput_best * 0.5)
            cmd = f"{cmd} --min-throughput={min_tput} -f minimize_latency.yml"

        print(f"  {cmd}")
        cmd_list = cmd.split(' ')
        result = subprocess.run(cmd_list, stdout=subprocess.PIPE)
        result_data = self._parse_experiment_output(result.stdout)

        if 'Percentile' not in result_data or 'Generator num configs' not in result_data or result_data[
                'Percentile'] == "None":
            print(f"{model} failed! Skipping")
            return None

        cmd = cmd.replace("main.py", "main.py -v")
        result = SweepResult(result_data, cmd, model)
        print(
            f"    {model}: Percentile = {result.percentile}, measurements = {result.num_measurements}"
        )
        return result

    def _analyze(self, key):
        results = self.results[key]
        percentiles = []
        below_cutoff = []
        num_measurements = []
        too_many_measurements = []
        too_few_measurements = []

        PERCENTILE_CUTOFF = 0.70
        NUM_MEASUREMENTS_MAX = 15
        NUM_MEASUREMENTS_MIN = 5

        for k, result in results.items():

            percentile = result.percentile
            num_measurement = result.num_measurements

            percentiles.append(percentile)
            num_measurements.append(num_measurement)

            if percentile < PERCENTILE_CUTOFF:
                below_cutoff.append(f"{percentile}: {result.cmd}")
            if num_measurement > NUM_MEASUREMENTS_MAX:
                too_many_measurements.append(f"{num_measurement}: {result.cmd}")
            if num_measurement < NUM_MEASUREMENTS_MIN:
                too_few_measurements.append(f"{num_measurement}: {result.cmd}")

        avg_percentile = mean(percentiles)
        median_percentile = median(percentiles)
        mode_percentile = mode(percentiles)

        print()
        print(
            f"{key} Average/Median/Mode percentiles = {avg_percentile:0.2f} / {median_percentile:0.2f} / {mode_percentile:0.2f}"
        )

        avg_measurements = mean(num_measurements)
        median_measurements = median(num_measurements)
        mode_measurements = mode(num_measurements)

        print(
            f"{key} Average/Median/Mode measurements = {avg_measurements:0.2f} / {median_measurements:0.2f} / {mode_measurements:0.2f}"
        )
        print()

        total_models = len(results)
        print(
            f"{key}: {len(below_cutoff)} out of {total_models} are below the cutoff of {100*PERCENTILE_CUTOFF}%:"
        )
        for x in sorted(below_cutoff)[:4]:
            print(f"  {x}")

        print()
        print()
        print(
            f"{key}: {len(too_many_measurements)} out of {total_models} are taking more than {NUM_MEASUREMENTS_MAX} measurements:"
        )
        for x in sorted(too_many_measurements, reverse=True)[:4]:
            print(f"  {x}")

        print()
        print()
        print(
            f"{key}: {len(too_few_measurements)} out of {total_models} are taking less than {NUM_MEASUREMENTS_MIN} measurements:"
        )
        for x in sorted(too_few_measurements)[:4]:
            print(f"  {x}")

    def cleanup_checkpoints(self, path):
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


sc = SweepCheckpoints()
sc.run()
