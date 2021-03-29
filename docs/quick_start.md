<!--
Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.

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

The steps below will guide you through using model analyzer to analyze a simple PyTorch model. If you are not using the docker installation, you may skip the first step. The instructions below assume a directory structure like the following:

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

Install Model Analyzer by following the instructions in the [Installation](docs/install.md) section, and run the Triton Model Analyzer container as shown below. 

```
$ docker run -it --privileged --rm --gpus all \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v $HOME/model_analyzer/examples/quick-start:/quick_start_repository \
        --net=host --name model-analyzer \
        model-analyzer /bin/bash
```

## Step 2: Run the `add_sub` example

The [examples/quick-start](../examples/quick-start) directory contains a simple libtorch model which calculates the sum and difference of two inputs. Run the Model Analyzer inside the container with:

```
$ mkdir analysis_results
$ model-analyzer -m /quick_start_repository -n add_sub --triton-launch-mode=local --export-path=analysis_results
```

You should see an output similar to the output below:

```
2021-03-17 01:14:01.397 INFO[entrypoint.py:288] Triton Model Analyzer started: config={'model_repository': '/quick_start_repository', 'model_names': [{'model_name': 'add_sub', 'objectives': {'perf_throughput': 10}, 'parameters': {'batch_sizes': [1], 'concurrency': []}}], 'objectives': {'perf_throughput': 10}, 'constraints': {}, 'batch_sizes': [1], 'concurrency': [], 'perf_analyzer_timeout': 600, 'perf_analyzer_cpu_util': 80.0, 'run_config_search_max_concurrency': 1024, 'run_config_search_max_instance_count': 5, 'run_config_search_disable': False, 'run_config_search_max_preferred_batch_size': 16, 'export': True, 'export_path': '.', 'summarize': True, 'filename_model_inference': 'metrics-model-inference.csv', 'filename_model_gpu': 'metrics-model-gpu.csv', 'filename_server_only': 'metrics-server-only.csv', 'max_retries': 100, 'duration_seconds': 5, 'monitoring_interval': 0.01, 'client_protocol': 'grpc', 'perf_analyzer_path': 'perf_analyzer', 'perf_measurement_window': 5000, 'perf_output': False, 'triton_launch_mode': 'local', 'triton_version': '21.02-py3', 'triton_http_endpoint': 'localhost:8000', 'triton_grpc_endpoint': 'localhost:8001', 'triton_metrics_url': 'http://localhost:8002/metrics', 'triton_server_path': 'tritonserver', 'triton_output_path': None, 'triton_server_flags': {}, 'log_level': 'INFO', 'gpus': ['all'], 'output_model_repository_path': './output_model_repository', 'override_output_model_repository': False, 'config_file': None, 'plots': [{'name': 'throughput_v_latency', 'title': 'Throughput vs. Latency', 'x_axis': 'perf_latency', 'y_axis': 'perf_throughput', 'monotonic': True}, {'name': 'gpu_mem_v_latency', 'title': 'GPU Memory vs. Latency', 'x_axis': 'perf_latency', 'y_axis': 'gpu_used_memory', 'monotonic': False}], 'top_n_configs': 3}
2021-03-17 01:14:01.419 INFO[entrypoint.py:102] Starting a local Triton Server...
2021-03-17 01:14:01.461 INFO[server_local.py:64] Triton Server started.
2021-03-17 01:14:03.477 INFO[driver.py:236] init
2021-03-17 01:14:06.375 INFO[server_local.py:81] Triton Server stopped.
2021-03-17 01:14:06.376 INFO[entrypoint.py:327] Starting perf_analyzer...
2021-03-17 01:14:06.376 INFO[analyzer.py:83] Profiling server only metrics...
2021-03-17 01:14:06.388 INFO[server_local.py:64] Triton Server started.
.
.
.
```

This will perform a search across various config parameters on the `add_sub` model. This takes over 40 minutes even on a TITAN RTX as Model Analyzer will try to find the search bounds automatically. When finished, it will generate the directory `./analysis_results` containing measured data and a summary report. The directory should be structured as follows. 

```
$HOME
  |--- model_analyzer
              |--- analysis_results
              .       |--- plots
              .       |      |--- add_sub
              .       |              |--- gpu_mem_v_latency.png
              .       |              |--- throughput_v_latency.png
                      | 
                      |--- results
                      |       |--- metrics-model-inference.csv 
                      |       |--- metrics-model-gpu.csv 
                      |       |--- metrics-server-only.csv
                      |
                      |--- reports
                              |--- add_sub
                                      |--- result_summary.pdf
```
