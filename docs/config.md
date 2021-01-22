<!--
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

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

# Model Analyzer Config

In addition to the CLI flags, you can config the Model Analyzer
using a [YAML](https://yaml.org/) file too. The scheme
for this configuration file is described below. Brackets
indicate that a parameter is optional. For non-list
and non-object parameters the value is set to the
specified default.

The placeholders below are used throughout the configuration:

* `<boolean>`: a boolean that can take `true` or `false` as value.
* `<string>`: a regular string
* `<comma-delimited-list>`: a list of comma separated items.
* `<int>`: a regular integer value.
* `<list>`: a list of values.
* `<range>`: An object containing `start` and `stop` key with an optional
`step` value. If `step` is not defined, we use 1 as the default value for
`step`. The types that support `<range>` can be described by a list or using
the example structure below:
```yaml
batch_sizes:
    start: 2
    stop: 6
    step: 2
```

This YAML represents the array `[2, 4, 6]`.

```yaml
# Path to the Model Repository
model_repository: <string>

# How Model Analyzer will launch triton. It should
# be either "docker", "local", or "remote".
# See docs/launch_modes.md for more information.
[ triton_launch_mode: <string> ]

# List of the model names to be analyzed
model_names: <comma-delimited-string|list>

# Concurrency values to be used for the analysis
[ concurrency: <comma-delimited-string|list|range> | default: 1 ]

# Batch size values to be used for the analysis
[ batch_sizes: <comma-delimited-string|list|range> | default: 1 ]

# Whether to export metrics to a file
[ export: <boolean> | default: false ]

# Export path to be used
[ export_path: <string> | default: '.' ]

# File name to be used for the model inference results
[ filename_model_inference: <string> | default: metrics-model-inference.csv ]

# File name to be used for the GPU metrics results
[ filename_model_gpu: <string> | default: metrics-model-gpu.csv ]

# File name to be used for storing the server only metrics.
[ filename_server_only: <string> | default: metrics-server-only.csv ]

# Specifies the maximum number of retries for any retry attempt.
[ max_retries: <int> | default: 100 ]

# Specifies how long (seconds) to gather server-only metrics
[ duration_seconds: <int> | default: 5 ]

# Duration of waiting time between each metric measurement in seconds
[ monitoring_interval: <float> | default: 0.01 ]

# The protocol used to communicate with the Triton Inference Server. Only 'http' and 'grpc' are allowed for the values.
[ client_protocol: <string> | default: grpc ]

# The full path to the perf_analyzer binary executable
[ perf_analyzer_path: <string> | default: perf_analzyer ]

# Time interval in milliseconds between perf_analyzer measurements.
# perf_analyzer will take measurements over all the requests completed within
# this time interval.
[ perf_measurement_window: <int> | default: 5000 ]

# Stops writing the output from the perf_analyzer to stdout.
[ no_perf_output: <bool> | default: false ]

# Triton Server version used when launching using Docker mode
[ triton_version: <string> | default: 20.11-py3 ]

# Logging level
[ log_level: <string> | default: INFO ]

# Triton Server HTTP endpoint url used by Model Analyzer client. Will be ignored if server-launch-mode is not 'remote'".
[ triton_http_endpoint: <string> | default: localhost:8000 ]

# Triton Server GRPC endpoint url used by Model Analyzer client. Will be ignored if server-launch-mode is not 'remote'".
[ triton_grpc_endpoint: <string> | default: localhost:8001 ]

# Triton Server metrics endpoint url used by Model Analyzer client. Will be ignored if server-launch-mode is not 'remote'".
[ triton_metrics_url: <string> | default: localhost:8002 ]

# The full path to the tritonserver binary executable
[ triton_server_path: <string> | default: tritonserver ]

# The full path to a file to write the Triton Server output log.
[ triton_output_path: <string> ]

# List of GPU UUIDs to be used for the profiling. Use 'all' to profile all the GPUs visible by CUDA."
[ gpus: <string|comma-delimited-list-string> | default: 'all' ]
```

If you save configuration file in `config.yml`, you can provide
the path to the Model Analyzer config using the 
`-f` flag.

```
model-analyzer -f config.yml
```

All the flags supported in the CLI, are supported in the configuration file
too.
 ## Example

An example configuration looks like below:

```yaml
model_repository: /path/to/model-repository
triton_launch_mode: docker
model_names:
  - vgg_19_graphdef

batch_sizes:
    start: 4
    stop: 9
concurrency:
    - 2
    - 4
    - 8
```

If you save this file to the `config.yml`, you can start
Model Analyzer using the `-f`, or `--config-file` flag.

This config will analyze the models using batch sizes equal
to `[4, 5, 6, 7, 8, 9]` and concurrency values equal to
`[2, 4, 8]`. You can also specify the `step` value
for batch size to change the step size.
