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

In addition to the CLI flags, you can config the Model Analyzer using a
[YAML](https://yaml.org/) file too. The scheme for this configuration file is
described below. Brackets indicate that a parameter is optional. For non-list
and non-object parameters the value is set to the specified default.

All the flags supported in the CLI, are supported in the configuration file
too. But some configurations are only supported through the config file.

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
* Other types are described in the end of this file.

```yaml
# Path to the Model Repository
model_repository: <string>

# How Model Analyzer will launch triton. It should
# be either "docker", "local", or "remote".
# See docs/launch_modes.md for more information.
[ triton_launch_mode: <string> ]

# List of the model names to be analyzed
model_names: <comma-delimited-string|list|model>

# Concurrency values to be used for the analysis
[ concurrency: <comma-delimited-string|list|range> | default: 1 ]

# Batch size values to be used for the analysis
[ batch_sizes: <comma-delimited-string|list|range> | default: 1 ]

# Whether to export metrics to a file
[ export: <boolean> | default: true ]

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

# Enables writing the output from the perf_analyzer to stdout.
[ perf_output: <bool> | default: false ]

# Triton Server version used when launching using Docker mode
[ triton_version: <string> | default: 21.02-py3 ]

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

# List of constraints placed on the config search results.
[ constrains: <constraint> ]

# List of objectives that user wants to sort the results by it.
[ objectives: <objective|list> ]

# Maximum concurrency used for the automatic config search.
[ run_config_search_max_concurrency: <int> | default: 1024 ]

# Maximum instance count used for the automatic config search.
[ run_config_search_max_instance_count: <int> | default: 5 ]

# Maximum instance count used for the automatic config search.
[ run_config_search_max_preferred_batch_size: <int> | default: 16 ]

# Disables automatic config search
[ run_config_search_disable: <bool> | default: false ]
```

## `<constraint>`
A constraint, specifies the bounds that determine a successful run. There are
three constraints allowed:

1. `perf_throughput`. Using this key you can specify the minimum desired throughput. The unit is
inference requests per second.
```yaml
perf_throughput:
    min: 5
```

The config above will filter out the results that their `perf_throughput` is less than
5 infer/sec.

2. `perf_latency`: Using this key you can specify the maximum tolerable latency or your latency budget. The unit here is milliseconds.
```yaml
perf_latency:
    max: 100
```

The config above will filter out the results that their `perf_latency` is larger than
100 milliseconds.

3. `gpu_used_memory`: Using this key you can specify the maximum GPU memory used by the model.
The unit for this field is megabytes.
```yaml
gpu_used_memory:
    max: 200
```

The config above will filter out the results that their `gpu_used_memory` is larger than
200 MBs.

You can also use these keys together:
```yaml
gpu_used_memory:
    max: 200
perf_latency:
    max: 100
```

This will filter out the results that their `gpu_used_memory` is larger than 200 MBs and
their latency is larger than 100 milliseconds.

The values described above can be specified both globally and on a per model basis.
The global example looks like below:

```yaml
model_repository: /path/to/model-repository
model_names:
  - model_1
  - model_2

constraints:
    gpu_used_memory:
        max: 200
```

In the global mode, the constraint specified will be enforced on every model.
If you have different constraints for each model, you can use the per model version:

```yaml
model_repository: /path/to/model-repository
model_names:
  model_1:
    constraints:
        gpu_used_memory:
            max: 200
  model_2:
    constraints:
        latency:
            max: 50
```

## `<objective>`

Objectives allow you to specify the sorting criteria for the final results.
The fields below are supported under this object type:

1. `perf_throughput`
2. `perf_latency`
3. `gpu_used_memory`
4. `gpu_free_memory`
5. `gpu_utilization`
6. `cpu_used_ram`
7. `cpu_available_ram`

An example `objectives` that will sort the results by throughput looks like below:

```yaml
objectives:
- perf_throughput
```

If you want to sort the results by latency, `objectives` should look like:

```yaml
objectives:
- perf_latency
```
### Weighted Objectives

In addition to the mode discussed above, you provide multiple values in
the objectives key. For example:

```yaml
objectives:
- perf_latency
- perf_throughput
```

The above config will multiply obtained throughput and latency for each
measurement by 0.5 (i.e. 0.5 * latency + 0.5 * throughput) and use this as
the "score" to sort the measurements according to. This mode can be useful
if you have a more advanced criteria upon which you want to sort the results.

An extension of the above `objectives` is explicitly specifying the weights.
For example:
```yaml
objectives:
    perf_latency: 10
    perf_throughput: 15
```

The score for each measurement will be a weighted average using the weights specified here.

Similar to `<constraint>`, `<objective>` can be specified both globally and on a
per model basis.

## `<parameter>`

This type can contain `concurrency` or `batch_sizes` as the possible keys. You
can specify both or one of them. `<parameter>` must be specified under the `model_names`
key for each model and cannot be specified globally.

An example `<parameter>` looks like below:

```yaml
parameters:
    concurrency:
        start: 2
        stop: 64
        step: 8
    batch_sizes: 1,2,3
```

A complete config example looks like below:

```yaml
model_repository: /path/to/model-repository
model_names:
  model_1:
    parameters:
        concurrency:
            start: 2
            stop: 64
            step: 8
        batch_sizes: 1,2,3
```

## `<model-config-parameters>`

This field represents the values that you want to change or sweep through
using Model Analyzer. All the values supported in the [Triton
Config](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md)
can be specified or swept through here. `<model-config-parameters>`  should be
specified on a per model basis and cannot be specified globally (like `<parameter>`).

An example `<model-config-parameters>` look like below:
```yaml
model_config_parameters:
    max_batch_size: [6, 8]
    dynamic_batching:
        preferred_batch_size: [[1], [2], [3]]
        max_queue_delay_microseconds: [200, 300]
    instance_group:
    -
        kind: KIND_GPU
        count: [1, 2]
```

Note that for values that accept a list by default you need to specify one
additional list if you want to sweep through it. Otherwise, it will be only
used to change the original model config to the value specified and it will
not sweep through it. You can look at the `preferred_batch_size` as an
example. This parameter is useful for manual config search. If you are
interested in the automatic config search, you can take a look at the [Config
Search](./config_search.md) docs.

A complete YAML config looks like below:
```yaml
model_repository: /path/to/model/repository/

run_config_search_disable: True
model_names:
  model_1:
    model_config_parameters:
        max_batch_size: 2
        dynamic_batching:
            preferred_batch_size: [1, 2, 3]
            max_queue_delay_microseconds: 200
        instance_group:
        -
            kind: KIND_GPU
            count: 1
        -
            kind: KIND_CPU
            count: 1
```

Note that in the above configuration, it will not sweep through any of the parameters. The
reason is that both `instance_group` and `preferred_batch_size` accept a list by default.
If you want to sweep through different parameters, you can use the configuration below:
```yaml
model_repository: /path/to/model/repository/

run_config_search_disable: True
model_names:
  model_1:
    model_config_parameters:
        max_batch_size: 2
        dynamic_batching:
            preferred_batch_size: [[1], [2], [3]]
            max_queue_delay_microseconds: 200
        instance_group:
        -
            -
                kind: KIND_GPU
                count: 1
        -
            -
                kind: KIND_CPU
                count: 1
```

This will lead to 6 different configurations (3 different preferred batch sizes and two instance group combinations).

## `<model>`
The model object can contain `<constraint>`, `<objective>`,
`<model-config-parameters>`, and `<parameter>`.

A model object puts together all the different parameters specified above. An example
will look like:
```yaml
model_repository: /path/to/model/repository/

run_config_search_disable: True
model_names:
  model_1:
    model_config_parameters:
        max_batch_size: 2
        dynamic_batching:
            preferred_batch_size: [[1], [2], [3]]
            max_queue_delay_microseconds: 200
        instance_group:
        -
            -
                kind: KIND_GPU
                count: 1
        -
            -
                kind: KIND_CPU
                count: 1
    parameters:
        concurrency:
            start: 2
            stop: 64
            step: 8
        batch_sizes: 1,2,3
    constraints:
        gpu_used_memory:
            max: 200
    objectives:
    - perf_throughput
```

You can specify multiple models under the `model_names` key too.

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

```
model-analyzer -f config.yml
```

This config will analyze the models using batch sizes equal
to `[4, 5, 6, 7, 8, 9]` and concurrency values equal to
`[2, 4, 8]`. You can also specify the `step` value
for batch size to change the step size.
