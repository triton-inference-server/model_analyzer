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

# Model Config Search

Model Analyzer's `profile` subcommand supports [Brute](config_search.md#brute-search-mode) and [Quick](config_search.md#quick-search-mode) search modes.

## Brute Search Mode

Model Analyzer's brute search mode supposts **automatic** and **manual**
sweeping through different configurations for Triton models.

### Automatic Configuration Search

Automatic configuration search is the default behavior when running Model
Analyzer. This mode is enabled when there is not any parameters specified for the
`model_config_parameters` section of the Model Analyzer Config. The parameters
that are automatically searched are
[`max_batch_size`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#maximum-batch-size)
and
[`instance_group`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#instance-groups).
Additionally, [`dynamic_batching`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#dynamic-batcher) will be enabled.

An example model analyzer config that performs automatic config search looks
like below:

```yaml
model_repository: /path/to/model/repository/

profile_models:
  - model_1
  - model_2
```

In the default mode, automatic config search will try values 1 through 5 for
[`instance_group`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#instance-groups).
The maximum value can be changed using the `run_config_search_max_instance_count` key in the Model Analyzer Config.
For each `instance_group`, Model Analyzer will sweep values 1 through 128 increasing exponentially (i.e. 1, 2, 4, 8, ...) for
[`max_batch_size`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#maximum-batch-size). The start and end values can be changed using `run_config_search_min_model_batch_size` and `run_config_search_max_model_batch_size`.
[`Dynamic_batching`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#dynamic-batcher)
will be enabled for all model configs generated using automatic search.

For each model config that is generated in automatic search, Model Analyzer will gather data for
[`concurrency`](https://github.com/triton-inference-server/server/blob/master/docs/perf_analyzer.md#request-concurrency)
values 1 through 1024 increased exponentially (i.e. 1, 2, 4, 8, ...). The maximum value can be configured
using the `run_config_search_max_concurrency` key in the Model Analyzer Config.

An example config that limits the search space used by Model Analyzer is
described below:

```yaml
model_repository: /path/to/model/repository/

run_config_search_max_instance_count: 3
run_config_search_max_concurrency: 8
profile_models:
  - model_1
  - model_2
```

If any `model_config_parameters` are specified for a model, it will disable
automatic searching of model configs and will only search within the values specified.
If `concurrency` is specified then only those values will be tried instead of the default concurrency sweep.
If both `concurrency` and `model_config_parameters` are specified, automatic
config search will be disabled.

For example, the config specified below will only automatically sweep through
the `model_config_parameters` that was described above:

```yaml
model_repository: /path/to/model/repository/

profile_models:
  model_1:
    concurrency: 1,2,3,128
```

The config described below will only sweep through different values for
`concurrency`:

```yaml
model_repository: /path/to/model/repository/

profile_models:
  model_1:
    model_config_parameters:
      instance_group:
        - kind: KIND_GPU
          count: [1, 2]
```

#### Important Note about Remote Mode

In the remote mode, `model_config_parameters` are always ignored because Model
Analyzer has no way of accessing the model repository of the remote Triton
Server. In this mode, only concurrency values can be swept.

### Manual Configuration Search

In addition to the automatic config search, Model Analyzer supports a manual
config search mode. To enable this mode, `--run-config-search-disable` flag
should be provided in the CLI or `run_config_search_disable: True` in the Model
Analyzer Config.

In this mode, values for both `concurrency` and `model_config_parameters` needs
to be specified. If no value for `concurrency` is specified, the default value,
1, will be used. This mode in comparison to the automatic mode, is not limited
to `max_batch_size`, `dynamic_batching`, and `instance_count` config parameters. Using manual
config search, you can create custom sweeps for every parameter that can be
specified in the model configuration. Model Analyzer only checks the syntax
of the `model_config_parameters` that is specified and cannot guarantee that
the configuration that is generated is loadable by Triton.

An example Model Analyzer Config that performs manual sweeping looks like below:

```yaml
model_repository: /path/to/model/repository/

run_config_search_disable: True
profile_models:
  model_1:
    model_config_parameters:
      max_batch_size: [6, 8]
      dynamic_batching:
        max_queue_delay_microseconds: [200, 300]
      instance_group:
        - kind: KIND_GPU
          count: [1, 2]
```

In this mode, Model Analyzer can sweep through every Triton model configuration
parameter available. For a complete list of parameters allowed under
`model_config_parameters`, refer to the [Triton Model
Configuration](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md).
It is your responsibility to make sure that the sweep configuration specified
works with your model. For example, in the above config, if we change `[6, 8]`
as the range for the `max_batch_size` to `[1]`, it will no longer be a valid
Triton Model Configuration.

The configuration sweep described above, will sweep through 8 configs = (2
`max_batch_size`) \* (2 `max_queue_delay_microseconds`) \* (2 `instance_group`) values.

### Examples of Additional Model Config Parameters

As mentioned in the previous section, manual configuration search allows you to
sweep on every parameter that can be specified in Triton model configuration. In
this section, we describe some of the parameters that might be of interest for
manual sweep:

- [Rate limiter](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#rate-limiter-config) setting
- If the model is using [ONNX](https://github.com/triton-inference-server/onnxruntime_backend) or [Tensorflow backend](https://github.com/triton-inference-server/tensorflow_backend), the "execution_accelerators" parameters. More information about this parameter is
  available in the [Triton Optimization Guide](https://github.com/triton-inference-server/server/blob/main/docs/optimization.md#framework-specific-optimization)

## Quick Search Mode

Quick search is enabled by adding the parameter `--run-config-search-mode quick` to the CLI.

It uses a hill climbing algorithm to search the configuration space, looking for
the maximal objective value within the specified constraints. In the majority of cases
this will find greater than 90% of the maximum objective value (that could be found using a brute force search),
while needing to search less than 10% of the configuration space.

It will then sweep the top-N configurations found (specified by `--num-configs-per-model`) over the default concurrency
range before generation of the summary reports.
