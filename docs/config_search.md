<!--
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

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

Model Analyzer's `profile` subcommand supports multiple modes when searching to find the best model configuration.

- [Brute](config_search.md#brute-search-mode) is the default, and will do a brute-force sweep of the cross product of all possible configurations
- [Quick](config_search.md#quick-search-mode) will use heuristics to try to find the optimal configuration much quicker than brute, and can be enabled via `--run-config-search-mode quick`
- [Multi-model](config_search.md#multi-model-search-mode) will profile mutliple models to find the optimal configurations for all models while they are running concurrently. This feature is enabled via `--run-config-profile-models-concurrently-enable`

## Brute Search Mode

Model Analyzer's brute search mode will do a brute-force sweep of the cross product of all possible configurations. You can [Manually](config_search.md#manual-brute-search) provide `model_config_parameters` to tell Model Analyzer what to sweep over, or you can
let it [Automatically](config_search.md#automatic-brute-search) sweep through configurations expected to have the highest impact on performance for Triton models.

### Automatic Brute Search

Automatic configuration search is the default behavior when running Model
Analyzer without manually specifying what values to search. The parameters
that are automatically searched are
[`max_batch_size`](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#maximum-batch-size)
and
[`instance_group`](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#instance-groups).
Additionally, [`dynamic_batching`](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#dynamic-batcher) will be enabled if it is legal to do so.

An example model analyzer config that performs automatic config search looks
like below:

```yaml
model_repository: /path/to/model/repository/

profile_models:
  - model_1
  - model_2
```

In the default mode, automatic config search will try values 1 through 5 for
[`instance_group`](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#instance-groups).
The maximum value can be changed using the `run_config_search_max_instance_count` key in the Model Analyzer Config.
For each `instance_group`, Model Analyzer will sweep values 1 through 128 increasing exponentially (i.e. 1, 2, 4, 8, ...) for
[`max_batch_size`](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#maximum-batch-size). The start and end values can be changed using `run_config_search_min_model_batch_size` and `run_config_search_max_model_batch_size`.
[`Dynamic_batching`](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#dynamic-batcher)
will be enabled for all model configs generated using automatic search.

For each model config that is generated in automatic search, Model Analyzer will gather data for
[`concurrency`](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/perf_analyzer.md#request-concurrency)
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
    parameters:
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

### Manual Brute Search

In addition to the automatic config search, Model Analyzer supports a manual
config search mode. To enable this mode, you can specify `model_config_parameters`
to sweep through, or set `--run-config-search-disable`

Unlike automatic mode, this mode is not limited to `max_batch_size`, `dynamic_batching`, and `instance_count` config parameters. Using manual
config search, you can create custom sweeps for every parameter that can be
specified in the model configuration. Model Analyzer only checks the syntax
of the `model_config_parameters` that is specified and cannot guarantee that
the configuration that is generated is loadable by Triton.

You can also specify `concurrency` ranges to sweep through. If unspecified, it will
automatically sweep concurrency for every model configuration (unless `--run-config-search-disable`
is set, in which case it will only use the concurrency value of 1)

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
Configuration](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md).
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

- [Rate limiter](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#rate-limiter-config) setting
- If the model is using [ONNX](https://github.com/triton-inference-server/onnxruntime_backend) or [Tensorflow backend](https://github.com/triton-inference-server/tensorflow_backend), the "execution_accelerators" parameters. More information about this parameter is
  available in the [Triton Optimization Guide](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md#framework-specific-optimization)

## Quick Search Mode

Quick search can be enabled by adding the parameter `--run-config-search-mode quick` to the CLI.

It uses a hill climbing algorithm to search the configuration space, looking for
the maximal objective value within the specified constraints. In the majority of cases
this will find greater than 95% of the maximum objective value (that could be found using a brute force search), while needing to search less than 10% of the configuration space.

After it has found the best config(s), it will then sweep the top-N configurations found (specified by `--num-configs-per-model`) over the default concurrency range before generation of the summary reports.

### Limiting Batch Size, Instance Count, and Client Concurrency

Using the `--run-config-search-<min/max>...` CLI options you have the ability to clamp the algorithm's upper or lower bounds for the model's batch size and instance count, as well as the client's concurrency.

Note: That by default quick search runs unbounded and ignores any default values for these settings

## Ensemble Model Search

_This mode has the following limitations:_

- Can only be run in `quick` search mode

Non-BLS Ensemble models can be optimized using the Quick Search mode's hill climbing algorithm to search the ensemble sub-model's configuration spaces in parallel, looking for the maximal objective value within the specified constraints. Model Analyzer has observed positive outcomes towards finding the maximum objective value; with runtimes under one hour (compared to the days it would take a brute force run to complete) for ensembles with up to four submodels.

After it has found the best config(s), it will then sweep the top-N configurations found (specified by `--num-configs-per-model`) over the concurrency range before generation of the summary reports.

## Multi-Model Search Mode

_This mode has the following limitations:_

- Can only be run in `quick` search mode
- Does not support detailed reporting, only summary reports

Multi-model concurrent search mode can be enabled by adding the parameter `--run-config-profile-models-concurrently-enable` to the CLI.

It uses Quick Search mode's hill climbing algorithm to search all models configurations spaces in parallel, looking for the maximal objective value within the specified constraints. Model Analyzer has observed positive outcomes towards finding the maximum objective value; with runtimes of around 20-30 minutes (compared to the days it would take a brute force run to complete).

After it has found the best config(s), it will then sweep the top-N configurations found (specified by `--num-configs-per-model`) over the default concurrency range before generation of the summary reports.

_Note:_ The algorithm attempts to find the most fair and optimal result for all models, by evaluating each model objective's gain/loss. In many cases this will result in the algorithm ranking higher a configuration that has a lower total combined throughput (if that was the objective), if this better balances the throughputs of all the models.

### Model Weighting

In additon to setting a model's objectives or constraints, in multi-model search mode, you have the ability to set a model's weighting. By default each model is set for equal weighting (value of 1), but in the YAML you can specify `weighting: <int>` which will bias that model's objectives when evaluating for an optimal result.

In the example below, the resnet50_libtorch model's objective gains (towards maximing latency) will have 3x the importance of the add_sub models throughput gains:

```yaml
profile_models:
  resnet50_libtorch:
    weighting: 3
    objectives:
      perf_latency_p99: 1
  add_sub:
    weighting: 1
    objectives:
      perf_throughput: 1
```
