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

import re, csv, sys, psutil, tempfile
from subprocess import Popen, STDOUT, TimeoutExpired
from statistics import mean
from itertools import product
from time import sleep
from pynvml import *

# This script will sweep through PA configurations, gather statistics, and then
# print out a csv to stdout. It does not take any options -- for now everything
# is specified at the top of the file under RUN CONFIGURATION
#
# REQUIREMENTS:
#
# - nvml python lib:
#     pip install nvidia-ml-py
#
# - A model repository at the below model_repository variable must exist with
#   the requested model name in it
#

### RUN CONFIGURATION ###

# Path to the model repository
model_repository = "output_model_repository"

# Name of the model to run (script only supports one model for now)
model_name = "resnet50_libtorch"

# This script will run the full cross product of all these listed options
pa_configurations = {
    "model": [model_name],
    "concurrency": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "batch_size": [1],
    "protocol": ["grpc", "http"],
    "is_async": [False, True]
}

# How long to wait between gathering statistics
measurement_interval = 0.1

# Timeout for each PA run
pa_timeout = 30

# Number of times to run each pa configuration
num_times = 1

# Switch to turn on/off printing of PA output
print_pa_output = False

##########################


class RunConfigData:
    """ Holds the configuration for one run """

    MEMBERS = [
        "model", "concurrency", "batch_size", "protocol", "is_async",
        "max_threads", "measurement_mode"
    ]

    def __init__(self,
                 model="ncf",
                 concurrency=1,
                 batch_size=1,
                 protocol="http",
                 is_async=False,
                 max_threads=16,
                 measurement_mode="count_windows"):
        self.model = model
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.protocol = protocol
        self.is_async = is_async
        self.max_threads = max_threads
        self.measurement_mode = measurement_mode

    def __repr__(self) -> str:
        str = ""
        for member in RunConfigData.MEMBERS:
            str += f"\n{member}: {getattr(self, member)}"
        return str

    def dict(self) -> dict:
        d = {}
        for member in RunConfigData.MEMBERS:
            d[member] = getattr(self, member)
        return d


class RunResultData:
    """ Holds the results for one run """
    MEMBERS = [
        "success", "time", "num_passes", "pa_cpu_usage", "pa_num_threads",
        "pa_mem_pct", "triton_cpu_usage", "triton_mem_pct", "gpu_utilization",
        "gpu_mem_pct", "throughput", "latency", "average_batch_size"
    ]

    def __init__(self):
        self.time = 0
        self.pa_cpu_usage = 0
        self.pa_num_threads = 0
        self.pa_mem_pct = 0
        self.triton_cpu_usage = 0
        self.triton_mem_pct = 0
        self.gpu_utilization = 0
        self.gpu_mem_pct = 0
        self.num_passes = 0
        self.success = False
        self.throughput = 0
        self.latency = 0
        self.average_batch_size = 0

    def dict(self) -> dict:
        d = {}
        for member in RunResultData.MEMBERS:
            d[member] = getattr(self, member)
        return d

    def __repr__(self) -> str:
        str = ""
        for member in RunResultData.MEMBERS:
            str += f"\n{member}: {getattr(self, member)}"
        return str


class PARunner:
    """ Runs PA and gathers GPU, CPU-PA, and CPU-Triton statistics """

    def __init__(self, triton_pid, gpu_handle, timeout, measurement_interval,
                 print_pa_output):
        self._pa_output = ""
        self._time = 0
        self._pa_cpu_usages = []
        self._pa_mem_pcts = []
        self._triton_cpu_usages = []
        self._triton_mem_pcts = []
        self._gpu_utilizations = []
        self._gpu_mem_pcts = []
        self._timeout = timeout
        self._print_pa_output = print_pa_output
        self._measurement_interval = measurement_interval
        self._run_result = RunResultData()
        self._triton_pid = triton_pid
        self._gpu_handle = gpu_handle

    def run_pa(self, cmd):
        self._reset()
        proc = self._create_pa_process(cmd)
        self._resolve_process(proc)
        self._update_run_result()

    def get_run_result(self):
        return self._run_result

    def _update_run_result(self):
        self._run_result.time = self._time
        self._run_result.pa_mem_pct = mean(
            self._pa_mem_pcts[5:]) if self._pa_mem_pcts[5:] else 0
        self._run_result.pa_cpu_usage = mean(
            self._pa_cpu_usages[5:]) if self._pa_cpu_usages[5:] else 0
        self._run_result.triton_cpu_usage = mean(
            self._triton_cpu_usages[5:]) if self._triton_cpu_usages[5:] else 0
        self._run_result.triton_mem_pct = mean(
            self._triton_mem_pcts[5:]) if self._triton_mem_pcts[5:] else 0
        self._run_result.gpu_utilization = mean(
            self._gpu_utilizations[5:]) if self._gpu_utilizations[5:] else 0
        self._run_result.gpu_mem_pct = mean(
            self._gpu_mem_pcts[5:]) if self._gpu_mem_pcts[5:] else 0
        r = re.findall(r'\[(\d+)\]', self._pa_output)
        if r:
            self._run_result.num_passes = int(r[-1])
        else:
            self._run_result.num_passes = 0
        r = re.search(
            'Concurrency: [0-9.e+]+, throughput: ([0-9.e+]+) infer/sec, latency ([0-9.e+]+) usec',
            self._pa_output)
        if r:
            self._run_result.success = True
            self._run_result.throughput = float(r.group(1))
            self._run_result.latency = float(r.group(2))
        else:
            self._run_result.success = False
            self._run_result.throughput = 0
            self._run_result.latency = 0

        if self._run_result.success:
            r1 = re.search('Inference count: ([0-9.e+]+)', self._pa_output)
            r2 = re.search('Execution count: ([0-9.e+]+)', self._pa_output)
            self._run_result.average_batch_size = float(r1.group(1)) / float(
                r2.group(1))

    def _reset(self):
        self._run_result = RunResultData()
        self._pa_output = ""
        self._time = 0
        self._pa_cpu_usages = []
        self._pa_mem_pcts = []
        self._triton_cpu_usages = []
        self._triton_mem_pcts = []
        self._gpu_mem_pcts = []
        self._gpu_utilizations = []

    def _create_pa_process(self, cmd):
        self._pa_log = tempfile.NamedTemporaryFile()
        try:
            process = Popen(cmd,
                            start_new_session=True,
                            stdout=self._pa_log,
                            stderr=STDOUT,
                            encoding='utf-8')
        except FileNotFoundError as e:
            raise Exception(f"command failed immediately: {e}")
        return process

    def _resolve_process(self, process):
        NUM_CORES = psutil.cpu_count()
        pa_process_util = psutil.Process(process.pid)
        triton_process_util = psutil.Process(self._triton_pid)

        while self._time < self._timeout:
            if process.poll() is not None:
                self._pa_output = self._get_process_output()
                if self._print_pa_output:
                    print(f"PA output is {self._pa_output}")
                break

            with pa_process_util.oneshot():
                pa_cpu_util = pa_process_util.cpu_percent() / NUM_CORES
                pa_num_threads = pa_process_util.num_threads()
                pa_mem_percent = pa_process_util.memory_percent()

            with triton_process_util.oneshot():
                triton_cpu_util = triton_process_util.cpu_percent() / NUM_CORES
                triton_mem_percent = triton_process_util.memory_percent()

            gpu_utilizations = nvmlDeviceGetUtilizationRates(gpu_handle)

            self._run_result.pa_num_threads = max(
                self._run_result.pa_num_threads, pa_num_threads)
            self._gpu_utilizations.append(gpu_utilizations.gpu)
            self._gpu_mem_pcts.append(gpu_utilizations.memory)
            self._pa_mem_pcts.append(pa_mem_percent)
            self._pa_cpu_usages.append(pa_cpu_util)
            self._triton_cpu_usages.append(triton_cpu_util)
            self._triton_mem_pcts.append(triton_mem_percent)

            self._time += self._measurement_interval
            sleep(self._measurement_interval)

        else:
            print('perf_analyzer took very long to exit, killing perf_analyzer')
            process.kill()

    def _get_process_output(self):
        self._pa_log.seek(0)
        tmp_output = self._pa_log.read()
        self._pa_log.close()
        return tmp_output.decode('utf-8')


class PATester():
    """ Primary class that runs through and tests PA configurations """

    def __init__(self, triton_pid, gpu_handle, timeout, measurement_interval,
                 print_pa_output):
        self._runner = PARunner(triton_pid=triton_pid,
                                gpu_handle=gpu_handle,
                                timeout=timeout,
                                measurement_interval=measurement_interval,
                                print_pa_output=print_pa_output)
        self._results = []

    def run(self, config: dict, num_tries: int):
        run_configs = self._get_configs(config)
        for run_config in run_configs:
            for _ in range(num_tries):
                self._run_config(run_config)

    def print_results(self):
        fieldnames = RunConfigData.MEMBERS + RunResultData.MEMBERS
        writer = csv.DictWriter(f=sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for config, result in self._results:
            dict1 = config.dict()
            dict2 = result.dict()
            dict1.update(dict2)
            for k, v in dict1.items():
                if isinstance(v, float):
                    dict1[k] = f'{v:0.1f}'
            writer.writerow(dict1)

    def _run_config(self, config):
        cmd = self._get_cmd(config)
        print(f"running {cmd}")
        self._runner.run_pa(cmd)
        results = self._runner.get_run_result()
        self._add_results(config, results)

    def _add_results(self, config, results):
        self._results.append((config, results))

    def _get_cmd(self, config: RunConfigData):
        # yapf: disable
        cmd = [
            "/usr/local/bin/perf_analyzer", "-v",
            "-i", config.protocol,
            "--measurement-mode", config.measurement_mode,
            "-m", config.model,
            "-b", str(config.batch_size),
            "--concurrency-range", str(config.concurrency),
            "--max-threads", str(config.max_threads)
        ]
        if config.protocol == "http":
            cmd += ["-u", "localhost:8000"]
        else:
            cmd += ["-u", "localhost:8001"]

        if config.is_async:
            cmd += ["--async"]
        else:
            cmd += ["--sync"]
        # yapf: enable
        return cmd

    def _get_dict_combos(self, config: dict):
        param_combinations = list(product(*tuple(config.values())))
        return [dict(zip(config.keys(), vals)) for vals in param_combinations]

    def _get_configs(self, config: dict):
        dict_combos = self._get_dict_combos(config)
        configs = []

        for c in dict_combos:
            config = RunConfigData()
            for k, v in c.items():
                setattr(config, k, v)
            configs.append(config)
        return configs


class TritonServer():
    """ Starts and stops a single triton server """

    def __init__(self):
        self._proc = None

    def start(self, model_repo, model):
        for proc in psutil.process_iter():
            try:
                # Check if process name contains the given name string.
                if "tritonserver" in proc.name().lower():
                    raise Exception("Tritonserver already running")
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                pass

        print(
            f"Starting tritonserver with repo={model_repo}, model={model_name}")
        cmd = self._get_cmd(model_repo, model)
        self._proc = self._create_process(cmd)
        sleep(5)
        if self._proc.poll() is not None:
            output = self._get_process_output()
            raise Exception(f"Tritonserver failed to launch: {output}")

        return self._proc.pid

    def stop(self):
        print(f"Stopping tritonserver")
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.communicate(timeout=10)
            except TimeoutExpired:
                self._proc.kill()
                self._proc.communicate()
        else:
            print(f"TritonServer does not exist?!")

    def _get_process_output(self):
        self._triton_log.seek(0)
        tmp_output = self._triton_log.read()
        self._triton_log.close()
        return tmp_output.decode('utf-8')

    def _create_process(self, cmd):
        self._triton_log = tempfile.NamedTemporaryFile()
        try:
            process = Popen(cmd,
                            start_new_session=True,
                            stdout=self._triton_log,
                            stderr=STDOUT,
                            encoding='utf-8')
        except FileNotFoundError as e:
            raise Exception(f"command failed immediately: {e}")
        return process

    def _get_cmd(self, model_repo, model):
        cmd = [
            "tritonserver", "--model-repository", model_repo, "--http-port",
            "8000", "--grpc-port", "8001", "--model-control-mode", "explicit",
            "--load-model", model
        ]
        return cmd


nvmlInit()
# FIXME hardcoded to 1 GPU
gpu_count = nvmlDeviceGetCount()
assert gpu_count == 1
gpu_handle = nvmlDeviceGetHandleByIndex(0)

server = TritonServer()
triton_pid = server.start(model_repository, model_name)

try:

    tester = PATester(triton_pid=triton_pid,
                      gpu_handle=gpu_handle,
                      timeout=pa_timeout,
                      measurement_interval=measurement_interval,
                      print_pa_output=print_pa_output)
    tester.run(pa_configurations, num_times)
    tester.print_results()
except Exception as e:
    print("Caught error. Stopping triton first")
    server.stop()
    nvmlShutdown()
    raise e

server.stop()
nvmlShutdown()
