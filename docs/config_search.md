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

# Table of Contents

- [Search Modes](#search-modes)
- [Default Search Mode](#default-search-mode)
- [Brute Search Mode](#brute-search-mode)
  - [Automatic Brute Search](#automatic-brute-search)
  - [Manual Brute Search](#manual-brute-search)
- [Quick Search Mode](#quick-search-mode)
- [Ensemble Model Search](#ensemble-model-search)
- [Multi-Model Search Mode](#multi-model-search-mode)

<br>

# Model Config Search

## Search Modes

Model Analyzer's `profile` subcommand supports multiple modes when searching to find the best model configuration:

- [Brute Force Search](config_search.md#brute-search-mode)
  - **Search type:** Brute-force sweep of the cross product of all possible configurations
  - **Default for:**
    - Single non-ensemble models
    - Multiple models being profiled sequentially
  - **Command:** `--run-config-search-mode brute`
- [Quick Search](config_search.md#quick-search-mode)
  - **Search type:** Heuristic sweep using a hill-climbing algorithm to find an optimal configuration
  - **Default for:**
    - Single ensemble models
    - Multiple models being profiled concurrently
  - **Command:** `--run-config-search-mode quick`

---

## Default Search Mode

Model Analyzer's default search mode depends on the type of model and if you are profiling models concurrently (in the case of multiple models):

- [Sequential (single or multi-model) Search](config_search.md#brute-search-mode)
  - **Default Search type:** [Brute Force Search](config_search.md#brute-search-mode)
  - **Command:** N/A
- [Concurrent / Multi-model Search](config_search.md#multi-model-search-mode)
  - **Default Search type:** [Quick Search](config_search.md#quick-search-mode)
  - **Command:** `--run-config-profile-models-concurrently-enable`
- [Ensemble Model Search](config_search.md#ensemble-model-search):
  - **Default Search type:** [Quick Search](config_search.md#quick-search-mode)
  - **Command:** N/A

---

## Brute Search Mode

**Default search mode when profiling non-ensemble models sequentially**

Model Analyzer's brute search mode will do a brute-force sweep of the cross product of all possible configurations. <br>
It has two modes:

- [Automatic](config_search.md#automatic-brute-search)
  - **No** model config parameters are specified
- [Manual](config_search.md#manual-brute-search)
  - **Any** model config parameters are specified
  - `--run-config-search-disable` option is specified

---

## Automatic Brute Search

**Default brute search mode when no model config parameters are specified**

The parameters that are automatically searched are
[model maximum batch size](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#maximum-batch-size),
[model instance groups](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#instance-groups), and [request concurrencies](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/perf_analyzer.md#request-concurrency).
Additionally, [dynamic_batching](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#dynamic-batcher) will be enabled if it is legal to do so.

_An example model analyzer YAML config that performs an Automatic Brute Search:_

```yaml
model_repository: /path/to/model/repository/

profile_models:
  - model_A
```

You can also modify the minimum/maximum values that the automatic search space will iterate through:

### [Instance Group Search Space](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#instance-groups)

- `Default:` 1 to 5 instance group counts
- `--run-config-search-min-instance-count: <val>`: Changes the instance group count's minimum automatic search space value
- `--run-config-search-max-instance-count: <val>`: Changes the instance group count's maximum automatic search space value

---

### [Model Batch Size Search Space](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md#maximum-batch-size)

- `Default:` 1 to 128 max batch sizes, sweeping over powers of 2 (i.e. 1, 2, 4, 8, ...)
- `--run-config-search-min-model-batch-size: <val>`: Changes the model's batch size minimum automatic search space value
- `--run-config-search-max-model-batch-size: <val>`: Changes the model's batch size maximum automatic search space value

---

### [Request Concurrency Search Space](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md#request-concurrency)

- `Default:` 1 to 1024 concurrencies, sweeping over powers of 2 (i.e. 1, 2, 4, 8, ...)
- `--run-config-search-min-concurrency: <val>`: Changes the request concurrency minimum automatic search space value
- `--run-config-search-max-concurrency: <val>`: Changes the request concurrency maximum automatic search space value

---

_An example YAML config that limits the search space:_

```yaml
model_repository: /path/to/model/repository/

run_config_search_max_instance_count: 3
run_config_search_min_model_batch_size: 16
run_config_search_min_concurrency: 8
run_config_search_max_concurrency: 256
profile_models:
  - model_A
```

_This will perform an Automatic Brute Search with instance group counts: 3-5, batch sizes: 16-128, and concurrencies: 8-256_

---

### **Interaction with Remote Triton Launch Mode**

When the triton launch mode is remote, _\*\*only concurrency values can be swept._\*\*<br>

Model Analyzer will ignore any model config parameters because we have no way of accessing and modifying the model repository of the remote Triton Server.

---

## Manual Brute Search

**Default brute search mode when any model config parameters or parameters are specified**

Using Manual Brute Search, you can create custom sweeps for any parameter that can be specified in the model configuration. There are two ways this mode is enabled when doing a brute search:

- Any [`model config parameters`](./config.md#model-config-parameters) are specified:
  - You must manually specify the batch sizes and instance group counts you want Model Analyzer to sweep
  - Request concurrencies will still be automatically swept (as they are not a model config parameter)
- Any [`parameters`](./config.md#test-configuration-parameter) are specified:
  - You must manually specify the batch sizes, instance group counts, and request concurrencies you want Model Analyzer to sweep
- `--run-config-search-disable` is specified:
  - You must manually specify the batch sizes, instance group counts, and request concurrencies you want Model Analyzer to sweep

_**Note**: Model Analyzer only checks the syntax of the `model config parameters` and cannot guarantee that the configuration that is generated is loadable by Triton. For a complete list of parameters allowed under model_config_parameters, refer to the [Triton Model
Configuration](https://github.com/triton-inference-server/server/blob/master/docs/user_guide/model_configuration.md). <br>**It is your responsibility to ensure the sweep configuration specified works with your model.**_

---

_In this example, Model Analyzer will only sweep through different values of `request concurrencies`:_

```yaml
model_repository: /path/to/model/repository/

profile_models:
  model_1:
    model_config_parameters:
      instance_group:
        - kind: KIND_GPU
          count: [1, 2]
```

_In this example, Model Analyzer will not sweep through any values, and will only try the concurrencies listed below:_

```yaml
model_repository: /path/to/model/repository/

profile_models:
  model_1:
    parameters:
      concurrency: 1,2,3,128
```

_This example shows how a value other than instance, batch size or concurrency can be swept, and will sweep through eight configs:<br>_

- \[2 \* `max_batch_size`] \* \[2 \* `max_queue_delay_microseconds`] \* \[2 \* `instance_group`].

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

---

### **Examples of Additional Model Config Parameters**

In this section, we describe some of the parameters that might be of interest for
manual sweep:

- [Rate limiter](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#rate-limiter-config) setting
- If the model is using [ONNX](https://github.com/triton-inference-server/onnxruntime_backend) or [Tensorflow backend](https://github.com/triton-inference-server/tensorflow_backend), the "execution_accelerators" parameters. More information about this parameter is
  available in the [Triton Optimization Guide](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md#framework-specific-optimization)

---

## Quick Search Mode

**Default search mode when profiling ensemble models or multiple models concurrently**

This mode uses a hill climbing algorithm to search the configuration space, looking for
the maximal objective value within the specified constraints. In the majority of cases
this will find greater than 95% of the maximum objective value (that could be found using a brute force search), while needing to search less than 10% of the configuration space.

After quick search has found the best config(s), it will then sweep the top-N configurations found (specified by `--num-configs-per-model`) over the default concurrency range before generation of the summary reports.

---

_An example model analyzer YAML config that performs a Quick Search:_

```yaml
model_repository: /path/to/model/repository/

run_config_search_mode: quick
profile_models:
  - model_A
```

---

### **Limiting Batch Size, Instance Group, and Client Concurrency**

Using the `--run-config-search-<min/max>...` config options you have the ability to clamp the algorithm's upper or lower bounds for the model's batch size and instance group count, as well as the client's request concurrency.

_Note: By default, quick search runs unbounded and ignores any default values for these settings_

---

_An example model analyzer YAML config that performs a Quick Search and constrains the search parameters:_

```yaml
model_repository: /path/to/model/repository/

run_config_search_mode: quick
run_config_search_max_instance_count: 4
run_config_search_max_concurrency: 16
run_config_search_min_model_batch_size: 8

profile_models:
  - model_A
```

---

## Ensemble Model Search

_This mode has the following limitations:_

- Can only be run in `quick` search mode
- Only supports up to 4 sub-models
- Does not support `cpu_only` option for submodels

Ensemble models can be optimized using the Quick Search mode's hill climbing algorithm to search the ensemble sub-model's configuration spaces in parallel, looking for the maximal objective value within the specified constraints. Model Analyzer has observed positive outcomes towards finding the maximum objective value; with runtimes under one hour (compared to the days it would take a brute force run to complete) for ensembles with up to four submodels.

After Model Analyzer has found the best config(s), it will then sweep the top-N configurations found (specified by `--num-configs-per-model`) over the concurrency range before generation of the summary reports.

---

## Multi-Model Search Mode

_This mode has the following limitations:_

- Can only be run in `quick` search mode
- Does not support detailed reporting, only summary reports

Multi-model concurrent search mode can be enabled by adding the parameter `--run-config-profile-models-concurrently-enable` to the CLI.

It uses Quick Search mode's hill climbing algorithm to search all models configurations spaces in parallel, looking for the maximal objective value within the specified constraints. Model Analyzer has observed positive outcomes towards finding the maximum objective value; with typical runtimes of around 20-30 minutes (compared to the days it would take a brute force run to complete) for a two to three model run.

After it has found the best config(s), it will then sweep the top-N configurations found (specified by `--num-configs-per-model`) over the default concurrency range before generation of the summary reports.

_Note:_ The algorithm attempts to find the most fair and optimal result for all models, by evaluating each model objective's gain/loss. In many cases this will result in the algorithm ranking higher a configuration that has a lower total combined throughput (if that was the objective), if this better balances the throughputs of all the models.

---

_An example model analyzer YAML config that performs a Multi-model search:_

```yaml
model_repository: /path/to/model/repository/

run_config_profile_models_concurrently_enable: true

profile_models:
  - model_A
  - model_B
```

---

### **Model Weighting**

In additon to setting a model's objectives or constraints, in multi-model search mode, you have the ability to set a model's weighting. By default each model is set for equal weighting (value of 1), but in the YAML you can specify `weighting: <int>` which will bias that model's objectives when evaluating for an optimal result.

---

_An example where model A's objective gains (towards minimizing latency) will have 3 times the importance versus maximizing model B's throughput gains:_

```yaml
model_repository: /path/to/model/repository/

run_config_profile_models_concurrently_enable: true

profile_models:
  model_A:
    weighting: 3
    objectives:
      perf_latency_p99: 1
  model_B:
    weighting: 1
    objectives:
      perf_throughput: 1
```
