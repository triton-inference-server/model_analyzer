<!--
Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# Table of Contents

- [Configuring Model Analyzer](#configuring-model-analyzer)
- [Config options for **profile**](#config-options-for-profile)
  - [CLI and YAML Config Options](#cli-and-yaml-config-options)
  - [YAML Only Options](#yaml-only-options)
- [Config Options for **report**](#config-options-for-report)
  - [CLI and YAML options](#cli-and-yaml-options)
  - [YAML only options](#yaml-only-options)
- [Field Descriptions](#field-descriptions)
- [Config Defaults](#config-defaults)

<br>

# Configuring Model Analyzer

Model Analyzer can be configured using either a [YAML](https://yaml.org/) config file, the `command line interface\* (CLI), or a combination of both.

- **Every** flag supported by Model Analyzer can be configured using a **YAML** config file
- Only a **subset** of flags can be configured using the **CLI**

---

The placeholders listed below are used throughout the configuration:

- `<boolean>`: a boolean that can take `true` or `false` as value
- `<string>`: a regular string
- `<comma-delimited-list>`: a list of comma separated items
- `<int>`: a regular integer value
- `<list>`: a list of values
- `<range>`: An object containing `start` and `stop` key with an optional `step`
  value
  - If `step` is not defined, **1** is the default step value
  - Types that support `<range>` can be described by a list or by using the example
    structure below, which declares the value of _batch_sizes_ to be an array `[2, 4, 6]`

```yaml
batch_sizes:
  start: 2
  stop: 6
  step: 2
```

- `<dict>`: a set of key-value pairs

```yaml
triton_server_flags:
  log_verbose: True
  exit_timeout_secs: 120
```

<br>

# Config options for profile

## CLI and YAML Config Options

A list of all the configuration options supported by **both the CLI and YAML**
config file are shown below.

- Brackets indicate that a parameter is optional.
- For non-list and non-object parameters the value is set to the specified default.
- The CLI flags corresponding to each of the options below are obtained by
  converting the `snake_case` options to `--kebab-case`.
  <br>
  For example, `profile_models` in the YAML would be `--profile-models` in the CLI.

```yaml
# Path to the Triton Model Repository
# https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md
model_repository: <string>

# List of the model names to be profiled
profile_models: <comma-delimited-string-list>

# Full path to directory to which to read and write checkpoints and profile data
[ checkpoint_directory: <string> | default: './checkpoints' ]

# The directory to which the model analyzer will save model config variants
[ output_model_repository_path: <string> | default: 'output_model_repository' ]

# Allow model analyzer to overwrite contents of the output model repository
[ override_output_model_repository: <boolean> | default: false ]

# Export path to be used
[ export_path: <string> | default: '.' ]

# Concurrency values to be used
[ concurrency: <comma-delimited-string|list|range> ]

# Batch size values to be used
[ batch_sizes: <comma-delimited-string|list|range> | default: 1 ]

# Specifies the maximum number of retries for any retry attempt
[ client_max_retries: <int> | default: 50 ]

# Specifies how long (seconds) to gather server-only metrics
[ duration_seconds: <int> | default: 3 ]

# Duration of waiting time between each metric measurement in seconds
[ monitoring_interval: <float> | default: 1 ]

# Specifies which metric(s) are to be collected.
[ collect_cpu_metrics: <bool> | default: false ]

# The protocol used to communicate with the Triton Inference Server. Only 'http' and 'grpc' are allowed for the values
[ client_protocol: <string> | default: grpc ]

# The full path to the perf_analyzer binary executable
[ perf_analyzer_path: <string> | default: perf_analyzer ]

# Perf analyzer timeout value in seconds
[ perf_analyzer_timeout: <int> | default: 600]

# Maximum CPU utilization value allowed for the perf_analyzer
[ perf_analyzer_cpu_util: <float> | default: 80.0 ]

# Enables writing the output from the perf_analyzer to a file or stdout
[ perf_output: <bool> | default: false ]

# If specified, setting --perf-output will write the perf_analyzer output to the file at # this location
[ perf_output_path: <str> ]

# Maximum number of times perf_analyzer is launched with auto adjusted parameters in an attempt to profile a model
[ perf_analyzer_max_auto_adjusts: <int> | default: 10 ]

# Disables model loading and unloading in remote mode
[ reload_model_disable: <bool> | default: false]

# Triton Docker image tag used when launching using Docker mode
[ triton_docker_image: <string> | default: nvcr.io/nvidia/tritonserver:23.02-py3 ]

# Triton Server HTTP endpoint url used by Model Analyzer client"
[ triton_http_endpoint: <string> | default: localhost:8000 ]

# The full path to the parent directory of 'lib/libtritonserver.so. Only required when using triton_launch_mode=c_api
[ triton_install_path: <string> | default: /opt/tritonserver ]

# Triton Server GRPC endpoint url used by Model Analyzer client
[ triton_grpc_endpoint: <string> | default: localhost:8001 ]

# Triton Server metrics endpoint url used by Model Analyzer client
[ triton_metrics_url: <string> | default: http://localhost:8002/metrics ]

# The full path to the tritonserver binary executable
[ triton_server_path: <string> | default: tritonserver ]

# The full path to a file to write the Triton Server output log
[ triton_output_path: <string> ]

# List of strings containing the paths to the volumes to be mounted into the tritonserver docker
# containers launched by model-analyzer. Will be ignored in other launch modes
[ triton_docker_mounts: <list of strings> ]

# The size of /dev/shm for the triton docker container
[ triton_docker_shm_size: <string>]

# How Model Analyzer will launch triton. It should
# be either "docker", "local", "remote" or "c_api".
# See docs/launch_modes.md for more information
[ triton_launch_mode: <string> | default: 'local' ]

# List of GPU UUIDs to be used for the profiling. Use 'all' to profile all the GPUs visible by CUDA
[ gpus: <string|comma-delimited-list-string> | default: 'all' ]

# Search mode. Options are "brute" and "quick"
[ run_config_search_mode: <string> | default: brute]

# Minimum concurrency used for the automatic config search
[ run_config_search_min_concurrency: <int> | default: 1 ]

# Maximum concurrency used for the automatic config search
[ run_config_search_max_concurrency: <int> | default: 1024 ]

# Minimum max_batch_size used for the automatic config search
[ run_config_search_min_model_batch_size: <int> | default: 1 ]

# Maximum max_batch_size used for the automatic config search
[ run_config_search_max_model_batch_size: <int> | default: 128 ]

# Minimum instance group count used for the automatic config search
[ run_config_search_min_instance_count: <int> | default: 1 ]

# Maximum instance group count used for the automatic config search
[ run_config_search_max_instance_count: <int> | default: 5 ]

# Disables automatic config search
[ run_config_search_disable: <bool> | default: false ]

# Enables the profiling of all supplied models concurrently
[ run_config_profile_models_concurrently_enable: <bool> | default: false]

# Skips the generation of summary reports and tables
[ skip_summary_reports: <bool> | default: false]

# Number of top configs to show in summary plots
[ num_configs_per_model: <int> | default: 3]

# Number of top model configs to save across ALL models, none saved by default
[ num_top_model_configs: <int> | default: 0 ]

# File name to be used for the model inference results
[ filename_model_inference: <string> | default: metrics-model-inference.csv ]

# File name to be used for the GPU metrics results
[ filename_model_gpu: <string> | default: metrics-model-gpu.csv ]

# File name to be used for storing the server only metrics
[ filename_server_only: <string> | default: metrics-server-only.csv ]

# Specifies columns keys for model inference metrics table
[ inference_output_fields: <comma-delimited-string-list> | default: See [Config Defaults](#config-defaults) section]

# Specifies columns keys for model gpu metrics table
[ gpu_output_fields: <comma-delimited-string-list> | default: See [Config Defaults](#config-defaults) section]

# Specifies columns keys for server only metrics table
[ server_output_fields: <comma-delimited-string-list> | default: See [Config Defaults](#config-defaults) section]

# Shorthand that allows a user to specify a max latency constraint in ms
[ latency_budget: <int>]

# Shorthand that allows a user to specify a min throughput constraint
[ min_throughput: <int>]

# Specify path to config YAML file
[ config_file: <string> ]
```

## YAML Only Options

The following config options are supported **only by the YAML** config file.

```yaml

# YAML config section for each model to be profiled
profile_models: <comma-delimited-string-list|list|profile_model>

# List of constraints placed on the config search results
[ constraints: <constraint> ]

# List of objectives that user wants to sort the results by it
[ objectives: <objective|list> ]

# Weighting used to bias the model's objectives (against the other models) in concurrent multi-model mode
[ weighting: <int>]

# Specify flags to pass to the Triton instances launched by model analyzer
[ triton_server_flags: <dict> ]

# Allows custom configuration of perf analyzer instances used by model analyzer
[ perf_analyzer_flags: <dict> ]

# Allows custom configuration of the environment variables for tritonserver instances
# launched by model analyzer
[ triton_server_environment: <dict> ]

# Dict of name=value pairs containing metadata for the tritonserver docker container
# launched in docker launch mode
[ triton_docker_labels: <dict> ]
```

<br>

# Config Options for report

## CLI and YAML options

A list of all the configuration options supported by **both the CLI and YAML**
config file are shown below.

- Brackets indicate that a parameter is optional.
- For non-list and non-object parameters the value is set to the
  specified default.

```yaml
# Comma-delimited list of the model names for which to generate detailed reports
report_model_configs: <comma-delimited-string-list>

# Full path to directory to which to read and write checkpoints and profile data
[ checkpoint_directory: <string> | default: '.' ]

# Export path to be used
[ export_path: <string> | default: '.' ]

# Specify path to config YAML file
[ config_file: <string> ]
```

## YAML only options

The following config options are support by the YAML config file only.

```yaml

# YAML config section for each model config for which to generate detailed reports
report_model_configs: <comma-delimited-string-list|list|report_model_config>

# YAML sections to configure the plots that should be shown in the detailed report
[ plots: <dict-plot-configs> | default: See [Config Defaults](#config-defaults) section ]
```

<br>

# Field Descriptions

Before proceeding, it will be helpful to see the documentation on [Model Analyzer Metrics](./metrics.md) regarding what metric tags are and how to use them.

### `<constraint>`

A constraint, specifies the bounds that determine a successful run. There are
three constraints allowed:

| Option Name        |   Units   | Constraint | Description                                          |
| :----------------- | :-------: | :--------: | :--------------------------------------------------- |
| `perf_throughput`  | inf / sec |    min     | Specify minimum desired throughput.                  |
| `perf_latency_p99` |    ms     |    max     | Specify maximum tolerable latency or latency budget. |
| `gpu_used_memory`  |    MB     |    max     | Specify maximum GPU memory used by model.            |

<br>

### Examples

---

To filter out the results when `perf_throughput` is less than 5 infer/sec:

```yaml
perf_throughput:
  min: 5
```

To filter out the results when `perf_latency_p99` is larger than 100 milliseconds:

```yaml
perf_latency_p99:
  max: 100
```

To filter out the results when `gpu_used_memory` is larger than 200 MBs:

```yaml
gpu_used_memory:
  max: 200
```

Keys can be combined for more complex constraints:

```yaml
gpu_used_memory:
  max: 200
perf_latency_p99:
  max: 100
```

This will filter out the results when `gpu_used_memory` is larger than 200 MBs
and their latency is larger than 100 milliseconds.

The values described above can be specified both globally and on a per model
basis.

The global example looks like below:

```yaml
model_repository: /path/to/model-repository
profile_models:
  - model_1
  - model_2

constraints:
  gpu_used_memory:
    max: 200
```

In the global mode, the constraint specified will be enforced on every model. To
have different constraints for each model, version below can be used:

```yaml
model_repository: /path/to/model-repository
profile_models:
  model_1:
    constraints:
      gpu_used_memory:
        max: 200
  model_2:
    constraints:
      perf_latency_p99:
        max: 50
```

---

### `<objective>`

Objectives specify the sorting criteria for the final results. The fields below
are supported under this object type:

| Option Name        | Description                                            |
| :----------------- | :----------------------------------------------------- |
| `perf_throughput`  | Use throughput as the objective.                       |
| `perf_latency_p99` | Use latency as the objective.                          |
| `gpu_used_memory`  | Use GPU memory used by the model as the objective.     |
| `gpu_free_memory`  | Use GPU memory not used by the model as the objective. |
| `gpu_utilization`  | Use the GPU utilization as the objective.              |
| `cpu_used_ram`     | Use RAM used by the model as the objective.            |
| `cpu_free_ram`     | Use RAM not used by the model as the objective.        |

An example `objectives` that will sort the results by throughput looks like
below:

```yaml
objectives:
  - perf_throughput
```

To sort the results by latency, `objectives` should look like:

```yaml
objectives:
  - perf_latency_p99
```

#### Weighted Objectives

In addition to the mode discussed above, multiple values can be provided in the
objectives key in order to provide more generalized control over how model
analyzer sorts results. For example:

```yaml
objectives:
  - perf_latency_p99
  - perf_throughput
```

The above config is telling model analyzer to compare two measurements by
finding relative gain from one measurement to the other, and computing the
weighted average of this gain across all listed metrics. In the above example,
the relative weights for each metric is equal by default. So if we have two
measurements of latency and throughput, model analyzer employs the following
logic:

```python
measurement_A = (latency_A, throughput_A)
measurement_B = (latency_B, throughput_B)

gain_A_B = (latency_A - latency_B, throughput_A - throughput_B)

weighted_average_gain = 0.5*(latency_A - latency_B) + 0.5*(throughput_A - throughput_B)
```

If `weighted_average_gain` exceeds a threshold then `measurement_A` is declared
to be "better" than `measurement_B`. Model Analyzer will automatically account
for metrics in which less is better and those which more is better.

An extension of the above `objectives` is explicitly specifying the weights. For
example:

```yaml
objectives:
  perf_latency_p99: 2
  perf_throughput: 3
```

The score for each measurement will be a weighted average using the weights
specified here. The above config will tell Model Analyzer to multiply the
throughput gain by `0.4` and latency gain by by `0.6`.

The `objectives` section can be specified both globally and on a per model
basis.

<br>

## Test Configuration `<parameter>`

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
profile_models:
  model_1:
    parameters:
      concurrency:
        start: 2
        stop: 64
        step: 8
      batch_sizes: 1,2,3
```

These parameters will result in testing the concurrency configurations of 2, 10,
18, 26, 34, 42, 50, 58, and 64, for each of different batch sizes of 1, 2 and 3.
This will result in 27 individual test runs of the model.
<br>

## `<weighting>`

This field is used to bias a model's objective when performing a multi-model search.

See [Multi-Model Search - Model Weighting](config_search.md#model-weighting) for details and an example YAML configuration.
<br>

## `<model-config-parameters>`

This field represents the values that can be changed or swept through using
Model Analyzer. All the values supported in the [Triton
Config](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md)
can be specified or swept through here. `<model-config-parameters>` should be
specified on a per model basis and cannot be specified globally (like
`<parameter>`).

Table below presents the list of common parameters that can be used for manual
sweeping:

|                                                                   Option                                                                    |                                                                                                                                                               Description                                                                                                                                                               |
| :-----------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [`dynamic_batching`](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#dynamic-batcher)  |                                                                                              Dynamic batching is a feature of Triton that allows inference requests to be combined by the server, so that a batch is created dynamically.                                                                                               |
| [`max_batch_size`](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#maximum-batch-size) |                                 The max_batch_size property indicates the maximum batch size that the model supports for the [types of batching](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/architecture.md#models-and-schedulers) that can be exploited by Triton.                                  |
|  [`instance_group`](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#instance-groups)   | Triton can provide multiple instances of a model so that multiple inference requests for that model can be handled simultaneously. The model configuration ModelInstanceGroup property is used to specify the number of execution instances that should be made available and what compute resource should be used for those instances. |

An example `<model-config-parameters>` look like below:

```yaml
model_config_parameters:
  max_batch_size: [6, 8]
  dynamic_batching:
    max_queue_delay_microseconds: [200, 300]
  instance_group:
    - kind: KIND_GPU
      count: [1, 2]
```

Note that for values that accept a list by default the user needs to specify an
additional list if want to sweep through it. Otherwise, it will only change the
original model config to the value specified and it will not sweep through it.
To read more about the automatic config search, check out
[Config Search](./config_search.md) docs.

A complete `profile` YAML config looks like below:

```yaml
model_repository: /path/to/model/repository/

run_config_search_disable: True
profile_models:
  model_1:
    model_config_parameters:
      max_batch_size: 2
      dynamic_batching:
        max_queue_delay_microseconds: 200
      instance_group:
        - kind: KIND_GPU
          count: 1
        - kind: KIND_CPU
          count: 1
```

Note that in the above configuration, it will not sweep through any of the
parameters. The reason is that `instance_group` accepts a list by default.
Sweeping thorough different parameters can be achieved
using the configuration below:

```yaml
model_repository: /path/to/model/repository/

run_config_search_disable: True
profile_models:
  model_1:
    model_config_parameters:
      max_batch_size: 2
      dynamic_batching:
        max_queue_delay_microseconds: [200, 400, 600]
      instance_group:
        - kind: KIND_GPU
          count: 1
        - kind: KIND_CPU
          count: 1
```

This will lead to 6 different configurations (3 different max queue delays
and two instance group count combinations). If both `model_config_parameters` and
`parameters` keys are specified, the list of sweep configurations will be the
cartesian product of both of the lists.
<br>

## `<cpu_only>`

This flag tells the model analyzer that, whether performing a search during profiling
or generating reports, this model should use CPU instances only. In order to run a model on CPU only you must provide a value of `true` for this flag.

```yaml
model_repository: /path/to/model/repository/
profile_models:
  model_1:
    cpu_only: true
  model_2:
    perf_analyzer_flags:
    percentile: 95
    latency-report-file: /path/to/latency/report/file
```

The above config tells model analyzer to profile `model_1` on CPU only,
but profile `model_2` using GPU.
<br>

## `<perf-analyzer-flags>`

This field allows the user to pass `perf_analyzer` any CLI options it needs to
execute properly. Refer to [the
`perf_analyzer`
docs](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md)
for more information on these options.

### Global options to apply to all instances of Perf Analyzer

---

The `perf_analyzer_flags` section can be specified globally to affect
perf_analyzer instances across all models in the following way:

```yaml
model_repository: /path/to/model/repository/
profile_models:
  model_1:
    parameters:
      batch_sizes: 4
perf_analyzer_flags:
  percentile: 95
  latency-report-file: /path/to/latency/report/file
```

### Model-specific options for Perf Analyzer

---

In order to set flags only for a specific model, you can specify
the flags in the following way:

```yaml
model_repository: /path/to/model/repository/
profile_models:
  model_1:
    perf_analyzer_flags:
      percentile: 95
      latency-report-file: /path/to/latency/report/file
```

### Shape, Input-Data, and Streaming

---

The `input-data`, `shape`, and `streaming` perf_analyzer options are additive and can take either
a single string (non-list) or a list of strings. Below is an example for `shape` argument:

```yaml
model_repository: /path/to/model/repository/
profile_models:
  model_1:
    parameters:
      batch_sizes: 4
perf_analyzer_flags:
  shape:
    - INPUT0:1024,1024
    - INPUT1:1024,1024
```

### Variable-sized dimensions

---

If a model configuration has variable-sized dimensions in the inputs section,
then the `shape` option of the `perf_analyzer_flags` option must be specified.
More information about this can be found in the
[Perf Analyzer documentation](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md#input-data).

### SSL Support:

---

Perf Analyzer supports SSL via GRPC and HTTP. It can be enabled via Model Analyzer configuration file updates.

**GRPC example**:

```yaml
model_repository: /path/to/model/repository/
profile_models:
  model_1:
    perf_analyzer_flags:
      ssl-grpc-root-certifications-file: /path/to/PEM/encoded/server/root/cert
```

**HTTPS example**:

```yaml
model_repository: /path/to/model/repository/
profile_models:
  model_1:
    perf_analyzer_flags:
      ssl-https-ca-certificates-file: /path/to/cert/authority/cert/file
      ssl-https-client-certificate-file: /path/to/client/cert/file
      ssl-https-private-key-file: /path/to/private/key/file
```

More information about this can be found in the
[Perf Analyzer documentation](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md#ssltls-support).

#### **Important Notes**:

- Only a subset of flags can be specified on the command line. Use `model-analyzer profile --help` to see the list of flags that can be specified on the command line. If a flag isn't listed there, it can be specified via the YAML config file.
- When providing arguments under `perf_analyzer_flags`, you must use `-` instead
  of `_`. This casing is important and Model Analyzer will not recognize
  `snake_cased` arguments.
- Model Analyzer also provides certain arguments to the `perf_analyzer`
  instances it launches. They are the following:
  - `concurrency-range`
  - `batch-size`
  - `model-name`
  - `measurement-mode`
  - `service-kind`
  - `triton-server-directory`
  - `model-repository`
  - `protocol`
  - `url`
    If provided under the `perf_analyzer_flags` section, their values will be overriden. Caution should therefore be exercised when overriding these.
    <br>

## `<triton-server-flags>`

This section of the config allows fine-grained control over the flags passed to
the Triton instances launched by Model Analyzer when it is running in the
`docker` or `local` Triton launch modes. Any argument to the server can be
specified here.

```yaml
model_repository: /path/to/model/repository/
profile_models:
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
  strict_model_config: False
  log_verbose: True
```

Since Model Analyzer relaunches Triton Server each time a model config is
loaded, you can also specify `triton_server_flags` on a per model basis. For
example:

```yaml
model_repository: /path/to/model/repository/
profile_models:
  model_1:
    triton_server_flags:
        strict_model_config: False
        log_verbose: True
  model_1:
    triton_server_flags:
        exit_timeout_secs: 120
```

**Note**:
Triton Server supports SSL via GRPC. It can be enabled via Model Analyzer configuration file updates.

**GRPC example:**

```yaml
model_repository: /path/to/model/repository/
profile_models:
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
  grpc-use-ssl: 1
  grpc-server-cert: /path/to/grpc/server/cert
  grpc-server-key: /path/to/grpc/server/keyfile
  grpc-root-cert: /path/to/grpc/root/cert
```

More information about this can be found in the
[Triton Server documentation](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inference_protocols.md#ssltls).

#### **Important Notes**:

- The Model Analyzer also provides certain arguments to the `tritonserver`
  instances it launches. These **_cannot_** be overridden by providing those
  arguments in this section. An example of this is `http-port`, which is an
  argument to Model Analyzer itself.
  <br>

## `<triton-server-environment>`

This section enables setting environment variables for the tritonserver
instances launched by Model Analyzer. For example, when a custom operation is
required by a model, Triton requires the LD_PRELOAD and LD_LIBRARY_PATH
environment variables to be set. See [this link](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/custom_operations.md)
for details. The value for this section is a dictionary where the
keys are the environment variable names and their values are the values to be
set.

```yaml
model_repository: /path/to/model/repository/
profile_models:
  - model_1
triton_server_environment:
  LD_PRELOAD: /path/to/custom/op/.so
  LD_LIBRARY_PATH: /path/to/shared/libararies
```

Since Model Analyzer relaunches Triton Server each time a model config is
loaded, you can also specify `triton_server_environment` on a per model basis. For
example:

```yaml
model_repository: /path/to/model/repository/
profile_models:
  model_1:
    triton_server_environment:
      LD_PRELOAD: /path/to/custom/op
```

**Important Notes**:

- The Model Analyzer also provides certain environment variables to the `tritonserver`
  instances it launches. These **_cannot_** be overridden by providing those
  arguments in this section. An example of this is `CUDA_VISIBLE_DEVICES`.
  <br>

## `<plots>`

This section is used to specify the kind of plots that will be displayed in the
detailed report. The section is structured as a list of `<plot>` objects as
follows:

```yaml
plots:
  plot_name_1:
    title: Title
    x_axis: perf_latency_p99
    y_axis: perf_throughput
    monotonic: True
  plot_name_2: .
    .
    .
```

Each `<plot>` object consists of the `plot_name` which is also the name of the
file to which the plot will be saved as a `.png`. Each plot object also requires
specifying each of the following:

- `title` : The title of the plot
- `x_axis` : The metric tag for the metric that should appear in the x-axis,
  e.g. `perf_latency_p99`. The plotted points are also sorted by the values of this
  metric.
- `y_axis` : The metric tag for the metric that should appear in the y-axis.
- `monotonic` : Some plots may require consecutive points to be strictly
  increasing. A boolean value of `true` can be specified here to require this.

<br>

## `<profile-model>`

The `--profile-models` argument can be provided as a list of strings (names of
models) from the CLI interface, or as a more complex `<profile-model>` object
but only through the YAML configuration file. The model object can contain
`<model-config-parameters>`, `<parameter>`,
`<perf-analyzer-flags>`,`<triton-server-flags>`,`<triton-server-environment>` and a flag `cpu_only`.

A profile model object puts together all the different parameters specified
above. An example will look like:

```yaml
model_repository: /path/to/model/repository/

run_config_search_disable: True
profile_models:
  model_1:
    perf_analyzer_flags:
      percentile: 95
      latency-report-file: /path/to/latency/report/file
    model_config_parameters:
      max_batch_size: 2
      dynamic_batching:
        max_queue_delay_microseconds: 200
      instance_group:
        - kind: KIND_GPU
          count: 1
        - kind: KIND_CPU
          count: 1
    parameters:
      concurrency:
        start: 2
        stop: 64
        step: 8
      batch_sizes: 1,2,3
```

Multiple models can be specified under the `profile_models` key.

#### **Example:**

```yaml
model_repository: /path/to/model-repository
run_config_search_disable: true
triton_launch_mode: docker
profile_models:
  vgg_19_graphdef:
    parameters:
      batch_sizes: 1, 2, 3
    model_config_parameters:
      dynamic_batching:
        max_queue_delay_microseconds: 200
      instance_group:
        - kind: KIND_CPU
          count: 1
    vgg_16_graphdef:
      parameters:
        concurrency:
          start: 2
          stop: 64
          step: 8
      model_config_parameters:
        dynamic_batching:
          max_queue_delay_microseconds: 200
        instance_group:
          - kind: KIND_GPU
            count: 1
batch_sizes:
  start: 4
  stop: 9
concurrency:
  - 2
  - 4
  - 8
```

If this file is saved to the `config.yml`, Model Analyzer can be started using
the `-f`, or `--config-file` flag.

```
model-analyzer -f /path/to/config.yml
```

It will run the model `vgg_19_graphdef` over combinations of batch sizes
`[1,2,3]`, `concurrency` `[2,4,8]` (taken from the global concurrency section),
with dynamic batching enabled and a single CPU instance.

It will also run the model `vgg_16_graphdef` over combinations of batch sizes
`[4,5,6,7,8,9]`(taken from the global `batch_sizes` section), concurrency
`[2,10,18,26,34,42,50,58]`, with dynamic batching enabled and a single GPU instance.
<br>

## `<report-model-config>`

The `--report-model-configs` argument can be provided as a list of strings
(names of models) from the CLI interface, or as a more complex
`<report-model-config>` object but only through the YAML configuration file. The
model object can contain `<plots>` objects. An example looks like:

```yaml
report_model_configs:
  model_config_0:
    plots:
      throughput_v_latency:
        title: Title
        x_axis: perf_latency_p99
        y_axis: perf_throughput
        monotonic: True
```

Multiple models can be specified under the `report_model_configs` key as well.

```yaml
report_model_configs:
  model_config_default:
    plots:
      throughput_v_latency:
        title: Title
        x_axis: perf_latency_p99
        y_axis: perf_throughput
        monotonic: True
  model_config_0:
    plots:
      gpu_mem_v_latency:
        title: Title
        x_axis: perf_latency_p99
        y_axis: gpu_used_memory
        monotonic: False
```

<br>

## Config Defaults

Up to date config defaults can be found in
[config_defaults.py](../model_analyzer/config/input/config_defaults.py)
