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

# FIXME -- currently assume triton is running

import re
import psutil
import tempfile
from subprocess import Popen, STDOUT
from statistics import mean
import multiprocessing


class RunConfig:

    def __init__(self,
                 model="ncf",
                 concurrency=1,
                 batch_size=1,
                 is_async=False,
                 measurement_mode="count_windows"):
        self.model = model
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.is_async = is_async
        self.measurement_mode = measurement_mode

    def __repr__(self) -> str:
        members = [
            attr for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        str = ""
        for member in members:
            str += f"\n{member}: {getattr(self, member)}"
        return str


class RunResult:

    def __init__(self):
        self.time = 0
        self.pa_cpu_usage = 0
        self.triton_cpu_usage = 0
        self.num_passes = 0
        self.success = False
        self.throughput = 0
        self.latency = 0

    def __repr__(self) -> str:
        members = [
            attr for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        str = ""
        for member in members:
            str += f"\n{member}: {getattr(self, member)}"
        return str


class PARunner:

    def __init__(self):
        self._pa_output = ""
        self._time = 0
        self._pa_cpu_usages = []
        self._triton_cpu_usages = []
        self._timeout = 30
        self._run_result = RunResult()

    def run_pa(self, cmd):
        self._reset()
        proc = self._create_pa_process(cmd)
        self._resolve_process(proc)
        self._update_run_result()

    def get_run_result(self):
        return self._run_result

    def _update_run_result(self):
        print(f"TKG: pa usage ignored was {self._pa_cpu_usages[5:]}")
        print(f"TKG: triton usage ignored was {self._triton_cpu_usages[5:]}")

        self._run_result.time = self._time
        self._run_result.pa_cpu_usage = mean(self._pa_cpu_usages[5:])
        self._run_result.triton_cpu_usage = mean(self._triton_cpu_usages[5:])
        r = re.findall(r'\[(\d+)\]', self._pa_output)
        if r:
            self._run_result.num_passes = int(r[-1])
        else:
            self._run_result.num_passes = 0
        r = re.search(
            'Concurrency: [0-9.]+, throughput: ([0-9.]+) infer/sec, latency ([0-9.]+) usec',
            self._pa_output)
        if r:
            self._run_result.success = True
            self._run_result.throughput = r.group(1)
            self._run_result.latency = r.group(2)
        else:
            self._run_result.success = False
            self._run_result.throughput = 0
            self._run_result.latency = 0

    def _reset(self):
        self._run_result = RunResult()
        self._pa_output = ""
        self._time = 0
        self._pa_cpu_usages = []
        self._triton_cpu_usages = []

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
        INTERVAL = 0.1
        pa_process_util = psutil.Process(process.pid)
        triton_process_util = psutil.Process(478)

        while self._time < self._timeout:
            if process.poll() is not None:
                self._pa_output = self._get_process_output()
                print(f"TKG: result is {self._pa_output}")
                break

            pa_cpu_util = pa_process_util.cpu_percent()
            triton_cpu_util = triton_process_util.cpu_percent(INTERVAL)
            self._pa_cpu_usages.append(pa_cpu_util)
            self._triton_cpu_usages.append(triton_cpu_util)
            self._time += INTERVAL

        else:
            print('perf_analyzer took very long to exit, killing perf_analyzer')
            process.kill()

    def _get_process_output(self):
        self._pa_log.seek(0)
        tmp_output = self._pa_log.read()
        self._pa_log.close()
        return tmp_output.decode('utf-8')


class TestPA():

    def __init__(self):
        self._runner = PARunner()

    def run(self, config: dict):
        cmds = self._get_cmds(config)
        for cmd in cmds:
            self._run_cmd(cmd)

    def _run_cmd(self, cmd):
        self._runner.run_pa(cmd)
        results = self._runner.get_run_result()
        print(f"TKG: results were {results}")

    def _get_cmd(self, config: RunConfig):
        cmd = [
            "/usr/local/bin/perf_analyzer", "-v", "-u", "localhost:8000", "-i",
            "http", "--measurement-mode", config.measurement_mode, "-m",
            config.model, "-b",
            str(config.batch_size), "--concurrency-range",
            str(config.concurrency)
        ]
        if config.is_async:
            cmd += ["--async"]
        else:
            cmd += ["--sync"]
        return cmd

    def _get_dict_combos(self, config: dict):
        from itertools import product
        param_combinations = list(product(*tuple(config.values())))
        return [dict(zip(config.keys(), vals)) for vals in param_combinations]

    def _get_cmds(self, config: dict):
        dict_combos = self._get_dict_combos(config)
        cmds = []

        for c in dict_combos:
            config = RunConfig()
            for k, v in c.items():
                setattr(config, k, v)
            cmd = self._get_cmd(config)
            cmds.append(cmd)
        return cmds


x = {"concurrency": [25, 50, 100, 200, 300, 400], "is_async": [True]}

main = TestPA()
main.run(x)
print(f"Num cores is {multiprocessing.cpu_count()}")
