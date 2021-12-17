<!--
Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Quick Start

The steps below will guide you through using model analyzer to analyze a simple PyTorch model.

## Step 1: Build and Run Model Analyzer Container

1. Clone the repository and build the docker:
```
$ git clone https://github.com/triton-inference-server/model_analyzer.git

$ cd ./model_analyzer

$ docker build --pull -t model-analyzer .
```

2. Run the docker:
```
$ docker run -it --rm --gpus all \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd)/examples/quick-start:/quick_start_repository \
    --net=host --name model-analyzer \
    model-analyzer /bin/bash
```

## Step 2: Profile the `add_sub` model

The [examples/quick-start](../examples/quick-start) directory contains a simple
libtorch model which calculates the sum and difference of two inputs. Run the
Model Analyzer `profile` subcommand inside the container with:

```
$ model-analyzer profile --model-repository /quick_start_repository --profile-models add_sub
```

If you already ran this earlier in the container, you can use the `--override-output-model-repository` option to overwrite the earlier results.

This will perform a search across limited configurable model parameters on the
`add_sub` model. This can take up to 60 minutes to finish. If you want a shorter
run (1-2 minutes) for example purposes, you can run with the below additional
options. Note that these options are not intended to find the best
configuration:

```
--run-config-search-max-concurrency 2 \
--run-config-search-max-instance-count 2 \
--run-config-search-preferred-batch-size-disable true
```

`--run-config-search-max-concurrency` sets the max concurrency value that run
config search should not go beyond. `--run-config-search-max-instance-count`
sets the max instance count value that run config search should not go beyond. `--run-config-search-preferred-batch-size-disable` disables the preferred batch
size search. With these options, model analyzer will test four configs. This
significantly reduces the search space, and therefore, model analyzer's runtime.

Here is some sample output:

```
...
2021-11-11 19:57:22.638 INFO[run_search.py:292] [Search Step] Concurrency set to 1. Instance count set to 1, and dynamic batching is disabled.
2021-11-11 19:57:22.662 INFO[server_local.py:99] Triton Server started.
2021-11-11 19:57:24.821 INFO[client.py:83] Model add_sub_i0 loaded.
2021-11-11 19:57:24.822 INFO[model_manager.py:221] Profiling model add_sub_i0...
2021-11-11 19:57:39.862 INFO[server_local.py:120] Stopped Triton Server.
2021-11-11 19:57:39.863 INFO[run_search.py:292] [Search Step] Concurrency set to 2. Instance count set to 1, and dynamic batching is disabled.
2021-11-11 19:57:39.883 INFO[server_local.py:99] Triton Server started.
2021-11-11 19:57:42.25 INFO[client.py:83] Model add_sub_i0 loaded.
2021-11-11 19:57:42.26 INFO[model_manager.py:221] Profiling model add_sub_i0...
2021-11-11 19:57:53.100 INFO[server_local.py:120] Stopped Triton Server.
2021-11-11 19:57:53.101 INFO[run_search.py:292] [Search Step] Concurrency set to 4. Instance count set to 1, and dynamic batching is disabled.
2021-11-11 19:57:53.121 INFO[server_local.py:99] Triton Server started.
2021-11-11 19:57:55.261 INFO[client.py:83] Model add_sub_i0 loaded.
2021-11-11 19:57:55.262 INFO[model_manager.py:221] Profiling model add_sub_i0...
2021-11-11 19:58:06.337 INFO[server_local.py:120] Stopped Triton Server.
...
```

**Note**: The checkpoint directory should be removed between consecutive runs of
the `model-analyzer profile` command.

## Generate tables and summary reports
In order to generate tables and summary reports, use the `analyze` subcommand as
follows.

```
$ mkdir analysis_results
$ model-analyzer analyze --analysis-models add_sub -e analysis_results
```

The measured data and summary report will be placed inside the
`./analysis_results` directory. The directory should be structured as follows.

```
$HOME
  |--- model_analyzer
              |--- analysis_results
              .       |--- plots
              .       |      |--- simple
              .       |      |      |--- add_sub
                      |      |              |--- gpu_mem_v_latency.png
                      |      |              |--- throughput_v_latency.png
                      |      |
                      |      |--- detailed
                      |             |--- add_sub
                      |                     |--- gpu_mem_v_latency.png
                      |                     |--- throughput_v_latency.png
                      | 
                      |--- results
                      |       |--- metrics-model-inference.csv 
                      |       |--- metrics-model-gpu.csv 
                      |       |--- metrics-server-only.csv
                      |
                      |--- reports
                              |--- summaries 
                              .        |--- add_sub
                              .                |--- result_summary.pdf
```

## Generate a detailed report

Model analyzer's report subcommand allows you to examine the performance of a
model config variant in detail. For example, it can show you the latency
breakdown of your model to help you identify potential bottlenecks in your model
performance. The detailed reports also contain other configurable plots and a
table of measurements taken of that particular config. You can generate a
detailed report for the two `add_sub` model configs `add_sub_i0` and
`add_sub_i1` using:

```
$ model-analyzer report --report-model-configs add_sub_i0,add_sub_i1 -e analysis_results 
```

This will create directories named after each of the model configs under
`.analysis_results/reports/detailed` containing the detailed report PDF files as
shown below.

```
$HOME
  |--- model_analyzer
              |--- analysis_results
              .       .
              .       .
                      .
                      |--- reports
                              .
                              .
                              .
                              |--- detailed
                                       |--- add_sub_i0
                                       |        |--- detailed_report.pdf
                                       |
                                       |--- add_sub_i1
                                                |--- detailed_report.pdf

```
