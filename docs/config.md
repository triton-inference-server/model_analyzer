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

# Configuring Model Analyzer

The Model Analyzer can be configured with a [YAML](https://yaml.org/) file or via the command line interface (CLI). Every flag supported by the CLI is supported in the configuration file, but some configurations are only supported using the config file.

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

The above YAML declares the value of batch_sizes to be an array `[2, 4, 6]`.

## CLI and YAML Config Options

A list of all the configuration options supported by both the CLI and YAML config file are shown below. Brackets indicate that a parameter is optional. For non-list and non-object parameters the value is set to the specified default.

The CLI flags corresponding to each of the options below are obtained by converting the `snake_case` options to `--kebab-case`. For example, `export_path` in the YAML would be `--export-path` in the CLI.


```yaml
# Path to the Model Repository
model_repository: <string>

# List of the model names to be analyzed
model_names: <comma-delimited-string>

# The directory to which the modela analyzer will save model config variants
[ output_model_repository_path: <string> | default: 'output_model_repository']

# Allow model analyzer to overwrite contents of the output model repository
[ override_output_model_repository: <boolean> | default: false ]

# Concurrency values to be used for the analysis
[ concurrency: <comma-delimited-string|list|range> | default: 1 ]

# Batch size values to be used for the analysis
[ batch_sizes: <comma-delimited-string|list|range> | default: 1 ]

# Whether to export metrics to a file
[ export: <boolean> | default: true ]

# Export path to be used
[ export_path: <string> | default: '.' ]

# Generate summary of results
[ summarize: <boolean>  | default: true ]

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

# Perf analyzer timeout value in seconds.
[ perf_analyzer_timeout: <int> | default: 600]

# Maximum CPU utilization value allowed for the perf_analyzer.
[ perf_analyzer_cpu_util: <float> | default: 80.0 ]

# Enables writing the output from the perf_analyzer to stdout.
[ perf_output: <bool> | default: false ]

# Triton Docker image tag used when launching using Docker mode
[ triton_docker_image: <string> | default: nvcr.io/nvidia/tritonserver:21.02-py3 ]

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

# How Model Analyzer will launch triton. It should
# be either "docker", "local", or "remote".
# See docs/launch_modes.md for more information.
[ triton_launch_mode: <string> ]

# Logging level
[ log_level: <string> | default: INFO ]

# List of GPU UUIDs to be used for the profiling. Use 'all' to profile all the GPUs visible by CUDA.
[ gpus: <string|comma-delimited-list-string> | default: 'all' ]

# Maximum concurrency used for the automatic config search.
[ run_config_search_max_concurrency: <int> | default: 1024 ]

# Maximum instance count used for the automatic config search.
[ run_config_search_max_instance_count: <int> | default: 5 ]

# Maximum instance count used for the automatic config search.
[ run_config_search_max_preferred_batch_size: <int> | default: 16 ]

# Disables automatic config search
[ run_config_search_disable: <bool> | default: false ]

# Specify path to config yaml file
[ config_file: <string> ]
```

## YAML Only Options

The following config options are supported only by the YAML config file.

```yaml

# List of the model names to be analyzed
model_names: <comma-delimited-string|list|model>

# List of constraints placed on the config search results.
[ constraints: <constraint> ]

# List of objectives that user wants to sort the results by it.
[ objectives: <objective|list> ]

# Specify flags to pass to the Triton instances launched by model analyzer
[ triton_server_flags: <dict> ]

```

## Field Descriptions

### `<constraint>`
A constraint, specifies the bounds that determine a successful run. There are
three constraints allowed:

| Option Name       |   Units   | Constraint | Description                                          |
| :---------------- | :-------: | :--------: | :--------------------------------------------------- |
| `perf_throughput` | inf / sec |    min     | Specify minimum desired throughput.                  |
| `perf_latency`    |    ms     |    max     | Specify maximum tolerable latency or latency budget. |
| `gpu_used_memory` |    MB     |    max     | Specify maximum GPU memory used by model.            |


#### Examples

To filter out the results when `perf_throughput` is
less than 5 infer/sec:

```yaml
perf_throughput:
    min: 5
```

To filter out the results when `perf_latency` is larger than 100
milliseconds:

```yaml
perf_latency:
    max: 100
```

To filter out the results when `gpu_used_memory` is larger than 200
MBs:
```yaml
gpu_used_memory:
    max: 200
```

Keys can be combined for more complex constraints:
```yaml
gpu_used_memory:
    max: 200
perf_latency:
    max: 100
```

This will filter out the results when `gpu_used_memory` is larger than 200 MBs and
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
To have different constraints for each model, version below can be used:

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

### `<objective>`

Objectives specify the sorting criteria for the final results.
The fields below are supported under this object type:

| Option Name       | Description                                            |
| :---------------- | :----------------------------------------------------- |
| `perf_throughput` | Use throughput as the objective.                       |
| `perf_latency`    | Use latency as the objective.                          |
| `gpu_used_memory` | Use GPU memory used by the model as the objective.     |
| `gpu_free_memory` | Use GPU memory not used by the model as the objective. |
| `gpu_utilization` | Use the GPU utilization as the objective.              |
| `cpu_used_ram`    | Use RAM used by the model as the objective.            |
| `cpu_free_ram`    | Use RAM not used by the model as the objective.        |

An example `objectives` that will sort the results by throughput looks like below:

```yaml
objectives:
- perf_throughput
```

To sort the results by latency, `objectives` should look like:

```yaml
objectives:
- perf_latency
```
#### Weighted Objectives

In addition to the mode discussed above, multiple values can be provided in
the objectives key in order to provide more generalized control over how model
analyzer sorts results. For example:

```yaml
objectives:
- perf_latency
- perf_throughput
```

The above config is telling model analyzer to compare two measurements by finding relative gain from one measurement to the other, and computing the weighted average of this gain across all listed metrics. In the above example, the relative weights for each metric is equal by default. So if we have two measurements of latency and throughput, model analyzer employs the following logic:

```python
measurement_A = (latency_A, throughput_A)
measurement_B = (latency_B, throughput_B)

gain_A_B = (latency_A - latency_B, throughput_A - throughput_B)

weighted_average_gain = 0.5*(latency_A - latency_B) + 0.5*(throughput_A - throughput_B)
```

If `weighted_average_gain` exceeds a threshold then `measurement_A` is declared to be "better" than `measurement_B`. Model Analyzer will automatically account for metrics in which less is better and those which more is better.

An extension of the above `objectives` is explicitly specifying the weights.
For example:
```yaml
objectives:
    perf_latency: 2
    perf_throughput: 3
```

The score for each measurement will be a weighted average using the weights
specified here. The above config will tell Model Analyzer to multiply the throughput gain by `0.4` and latency gain by by `0.6`.

The `objectives` section can be specified both globally and on a per model basis.

### Test Configuration `<parameter>`

A user can specify a range of test configurations that Model Analyzer will
profile over. The possible configuration parameters are `concurrency` and
`batch_sizes`. One or more parameters are specified per model only. Parameters
cannot be specified globally.

Options available under this parameter are described in table below:

| Option Name   | Description                                             | Supporting Types                                   |
| :------------ | :------------------------------------------------------ | :------------------------------------------------- |
| `concurrency` | Request concurrency used for generating the input load. | `<range>`, `<comma-delimited-list>`, or a `<list>` |
| `batch_sizes` | Static batch size used for generating requests.         | `<range>`, `<comma-delimited-list>`, or a `<list>` |


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

These parameters will result in testing the concurrency configurations of 2,
10, 18, 26, 34, 42, 50, 58, and 64, for each of different batch sizes of 1, 2
and 3. This will result in 27 individual test runs of the model.

### `<model-config-parameters>`

This field represents the values that can be changed or swept through
using Model Analyzer. All the values supported in the [Triton
Config](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md)
can be specified or swept through here. `<model-config-parameters>`  should be
specified on a per model basis and cannot be specified globally (like `<parameter>`).

Table below presents the list of common parameters that can be used for manual sweeping:

|                                                              Option                                                              |                                                                                                                                                               Description                                                                                                                                                               |
| :------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [`dynamic_batching`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#dynamic-batcher)  |                                                                                              Dynamic batching is a feature of Triton that allows inference requests to be combined by the server, so that a batch is created dynamically.                                                                                               |
| [`max_batch_size`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#maximum-batch-size) |                                       The max_batch_size property indicates the maximum batch size that the model supports for the [types of batching](https://github.com/triton-inference-server/server/blob/master/docs/architecture.md#models-and-schedulers) that can be exploited by Triton.                                       |
|  [`instance_group`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#instance-groups)   | Triton can provide multiple instances of a model so that multiple inference requests for that model can be handled simultaneously. The model configuration ModelInstanceGroup property is used to specify the number of execution instances that should be made available and what compute resource should be used for those instances. |

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

Note that for values that accept a list by default the user needs to specify
an additional list if want to sweep through it. Otherwise, it will only
change the original model config to the value specified and it will not sweep
through it. `preferred_batch_size` is an example for this. To read more about
the automatic config search, checkout [Config Search](./config_search.md)
docs.

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
Sweeping thorough different parameters can be achieved using the configuration below:
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

This will lead to 6 different configurations (3 different preferred batch
sizes and two instance group combinations). If both
`model_config_parameters` and `parameters` keys are specified, the list of sweep
configurations will be the cartesian product of both of the lists.

### `<perf-analyzer-flags>`

This field allows fine-grained control over the behavior of the `perf_analyzer` instances launched by Model Analyzer. `perf_analyzer` options and their values can be specified here, and will be passed to `perf_analyzer`. Refer to [the `perf_analyzer` docs](https://github.com/triton-inference-server/server/blob/master/docs/perf_analyzer.md) for more information on these options.

#### Example

```yaml
model_repository: /path/to/model/repository/
model_names:
  model_1:
    perf_analyzer_flags:
        percentile: 95
        latency-report-file: /path/to/latency/report/file
```
**Important Notes**: 
1. The section name contains underscores `perf_analyzer_flags` as seen in the example above. However the arguments themselves should contain hyphens as seen with `latency-report-file`.
2. The Model Analyzer also provides certain arguments to the `perf_analyzer` instances it launches. These ***cannot*** be overriden by providing those arguments in this section. An example of this is `perf_measurement_window`, which is an argument to Model Analyzer itself.

### The `model-names` field and `<model>`
The `--model-names` argument can be provided as a list of strings (names of models) from the CLI interface, or as a more complex `<model>` object but only through the YAML configuration file. The model object can contain `<constraint>`, `<objective>`,
`<model-config-parameters>`, `<parameter>` and `<perf-analyzer-flags>`.

A model object puts together all the different parameters specified above. An example
will look like:
```yaml
model_repository: /path/to/model/repository/

run_config_search_disable: True
model_names:
  model_1:
    perf_analyzer_flags:
        percentile: 95
        latency-report-file: /path/to/latency/report/file
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

Multiple models can be specified under the `model_names` key.

 #### Example

An example configuration looks like below:

```yaml
model_repository: /path/to/model-repository
run_config_search_disable: true
triton_launch_mode: docker
model_names:
  vgg_19_graphdef:
    parameters:
        batch_sizes: 1, 2, 3
    model_config_parameters:
        dynamic_batching:
            preferred_batch_size: [[2], [3]]
            max_queue_delay_microseconds: 200
        instance_group:
        -
            -
                kind: KIND_CPU
                count: 1
    vgg_16_graphdef:
      parameters:
        concurrency:
            start: 2
            stop: 64
            step: 8
      model_config_parameters:
        dynamic_batching:
            preferred_batch_size: [[1], [2]]
            max_queue_delay_microseconds: 200
        instance_group:
        -
            -
                kind: KIND_GPU
                count: 1
batch_sizes:
    start: 4
    stop: 9
concurrency:
    - 2
    - 4
    - 8
```

If this file is saved to the `config.yml`, 
Model Analyzer can be started using the `-f`, or `--config-file` flag.

```
model-analyzer -f config.yml
```

It will run the model `vgg_19_graphdef` over combinations of batch sizes `[1,2,3]`, `concurrency` `[2,4,8]` (taken from the global concurrency section), with dynamic batching enabled and preferred batch sizes `2` and `3` and a single CPU instance.

It will also run the model `vgg_16_graphdef` over combinations of batch sizes `[4,5,6,7,8,9]`(taken from the global `batch_sizes` section), concurrency `[2,10,18,26,34,42,50,58]`, with dynamic batching enabled and preferred batch sizes `1` and `2` and a single GPU instance.

### `<triton-server-flags>`

This section of the config allows fine-grained control over the flags passed to the Triton instances launched by Model Analyzer when it is running in the `docker` or `local` Triton launch modes. Any argument to the server can be specified here.

#### Example

```yaml
model_repository: /path/to/model/repository/
model_names:
  model_1:
    parameters:
        batch_sizes:
            start: 4
            stop: 9
        concurrency:
            - 2
            - 4
            - 8
triton_server_flags:
    strict-model-config: False
    log-verbose: True
```

**Important Notes**: 
1. The section name contains underscores `triton_server_flags` as seen in the example above. However the arguments themselves should contain hyphens as seen with `strict-model-config`.
2. The Model Analyzer also provides certain arguments to the `tritonserver` instances it launches. These ***cannot*** be overriden by providing those arguments in this section. An example of this is `http-port`, which is an argument to Model Analyzer itself.

