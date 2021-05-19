<!--
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

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

The steps below will guide you through using model analyzer to analyze a simple
PyTorch model. If you are not using the docker installation, you may skip the
first step. The instructions below assume a directory structure like the
following:

```
$HOME
  |--- model_analyzer
              |--- docs
              |--- examples
              |--- helm-chart
              |--- images
              |--- model_analyzer
              |--- qa
              |--- tests
              .
              .
              .
```

## Step 1: Install Model Analyzer and Run Container

Install Model Analyzer by following the instructions in the
[Installation](./install.md) section, and run the Triton Model Analyzer
container as shown below. 

```
$ docker run -it --rm --gpus all \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v $HOME/model_analyzer/examples/quick-start:/quick_start_repository \
        --net=host --name model-analyzer \
        model-analyzer /bin/bash
```

## Step 2: Profile the `add_sub` model

The [examples/quick-start](../examples/quick-start) directory contains a simple
libtorch model which calculates the sum and difference of two inputs. Run the
Model Analyzer `profile` subcommand inside the container with:

```
$ model-analyzer profile -m /quick_start_repository/ --profile-models add_sub
```

If a directory called `./output_model_repository` already exists, you will
receive the following error:

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/model_analyzer/entrypoint.py", line 278, in create_output_model_repository
    os.mkdir(config.output_model_repository_path)
FileExistsError: [Errno 17] File exists: './output_model_repository'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/bin/model-analyzer", line 8, in <module>
    sys.exit(main())
  File "/usr/local/lib/python3.8/dist-packages/model_analyzer/entrypoint.py", line 307, in main
    create_output_model_repository(config)
  File "/usr/local/lib/python3.8/dist-packages/model_analyzer/entrypoint.py", line 281, in create_output_model_repository
    raise TritonModelAnalyzerException(
model_analyzer.model_analyzer_exceptions.TritonModelAnalyzerException: Path "./output_model_repository" already exists. Please set or modify "--output-model-repository-path" flag or remove this directory. You can also allow overriding of the output directory using the "--override-output-model-repository" flag.
```
This is to give you a chance to save/move any model configs you may have from a
previous run in the output repository. Simply add the
`--override-output-model-repository` flag to tell model analyzer it can safely
delete the contents of the directory.

```
$ model-analyzer profile -m /quick_start_repository/ --profile-models add_sub --override-output-model-repository
```
You should see an output similar to the output below:

```

2021-05-11 02:02:58.194 INFO[entrypoint.py:98] Starting a local Triton Server...
2021-05-11 02:02:58.203 INFO[server_local.py:64] Triton Server started.
2021-05-11 02:03:02.609 INFO[server_local.py:81] Triton Server stopped.
2021-05-11 02:03:02.610 INFO[analyzer_state_manager.py:118] No checkpoint file found, starting a fresh run.
2021-05-11 02:03:02.610 INFO[analyzer.py:82] Profiling server only metrics...
2021-05-11 02:03:02.618 INFO[server_local.py:64] Triton Server started.
2021-05-11 02:03:05.766 INFO[gpu_monitor.py:72] Using GPU(s) with UUID(s) = { GPU-e35ba3d2-6eef-2bb9-e35c-6ef6eada4f11 } for profiling.
2021-05-11 02:03:08.720 INFO[server_local.py:81] Triton Server stopped.
2021-05-11 02:03:08.721 INFO[run_search.py:146] Will sweep both the concurrency and model config parameters...
2021-05-11 02:03:08.721 INFO[run_search.py:289] Concurrency set to 1. Instance count set to 1, and dynamic batching is disabled.
2021-05-11 02:03:08.736 INFO[server_local.py:64] Triton Server started.
2021-05-11 02:03:14.512 INFO[client.py:80] Model add_sub_i0 loaded.
2021-05-11 02:03:14.514 INFO[model_manager.py:211] Profiling model add_sub_i0...
2021-05-11 02:03:16.578 INFO[gpu_monitor.py:72] Using GPU(s) with UUID(s) = { GPU-e35ba3d2-6eef-2bb9-e35c-6ef6eada4f11 } for profiling.
.
.
.
```

This will perform a search across various config parameters on the `add_sub`
model. This takes over 40 minutes even on a TITAN RTX as Model Analyzer will try
to find the search bounds automatically. When finished, Model analyzer stores
all of the profiling measurements it has taken in a binary file in the checkpoint directory (See [config_defaults.py](../model_analyzer/config/input/config_defaults.py) for default location).

```
$ ls -l checkpoints
total 12
-rw-r--r-- 1 root root 11356 May 11 13:00 0.ckpt
```

Refer to the [checkpoints](./checkpoints.md) documentation for more details on
how checkpoint files work. 

## Generate tables and summary reports
In order to generate tables and summary reports, use the `analyze` subcommand as
follows.

```
$ mkdir analysis results
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
