<!--
Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
--run-config-search-max-model-batch-size 2 \
--run-config-search-max-instance-count 2
```

`--run-config-search-max-concurrency` sets the max concurrency value that run
config search will not go beyond. `--run-config-search-max-model-batch-size` sets the highest max_batch_size that run config search will not go beyond.  `--run-config-search-max-instance-count`
sets the max instance count value that run config search will not go beyond. With these options, model analyzer will test 5 configs (4 new configs as well as the unmodified default add_sub config), and each config will have 2 experiments run on Perf Analyzer (concurrency=1 and concurrency=2). This significantly reduces the search space, and therefore, model analyzer's runtime.

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
detailed report for the two `add_sub` model configs `add_sub_config_default` and
`add_sub_config_0` using:

```
$ model-analyzer report --report-model-configs add_sub_config_default,add_sub_config_0 -e analysis_results 
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
                                       |--- add_sub_config_default
                                       |        |--- detailed_report.pdf
                                       |
                                       |--- add_sub_config_0
                                                |--- detailed_report.pdf

```
