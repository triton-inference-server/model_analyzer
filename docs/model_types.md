<!--
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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

- [Multi-Model](#multi-model)
- [Ensemble](#ensemble)
- [BLS](#bls)
- [LLM](#llm)

<br>

## Support Matrix

The table below illustrates shows which search modes are available based on the model type:

|            | Single/Multi-Model | BLS | Ensemble | LLM |
| ---------- | ------------------ | --- | -------- | --- |
| **Brute**  | o                  | -   | -        | -   |
| **Quick**  | o                  | o   | o        | o   |
| **Optuna** | o                  | o   | o        | o   |

## Multi-Model

_This mode has the following limitations:_

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

In addition to setting a model's objectives or constraints, in multi-model search mode, you have the ability to set a model's weighting. By default each model is set for equal weighting (value of 1), but in the YAML you can specify `weighting: <int>` which will bias that model's objectives when evaluating for an optimal result.

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

## Ensemble

_Profiling this model type has the following limitations:_

- Only supports up to four composing models
- Composing models cannot be ensemble or BLS models

Ensemble models can be optimized using the Quick Search mode's hill climbing algorithm to search the composing models' configuration spaces in parallel, looking for the maximal objective value within the specified constraints. Model Analyzer has observed positive outcomes towards finding the maximum objective value; with runtimes under one hour (compared to the days it would take a brute force run to complete) for ensembles that contain up to four composing models.

After Model Analyzer has found the best config(s), it will then sweep the top-N configurations found (specified by `--num-configs-per-model`) over the concurrency range before generation of the summary reports.

---

## BLS

_Profiling this model type has the following limitations:_

- Only supports up to four composing models
- Composing models cannot be ensemble or BLS models

BLS models can be optimized using the Quick Search mode's hill climbing algorithm to search the BLS composing models' configuration spaces, as well as the BLS model's instance count, in parallel, looking for the maximal objective value within the specified constraints. Model Analyzer has observed positive outcomes towards finding the maximum objective value; with runtimes under one hour (compared to the days it would take a brute force run to complete) for BLS models that contain up to four composing models.

After Model Analyzer has found the best config(s), it will then sweep the top-N configurations found (specified by `--num-configs-per-model`) over the concurrency range before generation of the summary reports.

---

## LLM

_Profiling this model type has the following limitations:_

- Summary/Detailed reports do not include the new metrics

In order to profile LLMs you must tell MA that the model type is LLM by setting `--model-type LLM` in the CLI/config file. You can specify CLI options to the GenAI-Perf tool using `genai_perf_flags`. See the [GenAI-Perf CLI](https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/genai-perf/README.md#cli) documentation for a list of the flags that can be specified.

LLMs can be optimized using either Quick or Brute search mode.

_An example model analyzer YAML config for a LLM:_

```yaml
model_repository: /path/to/model/repository/

model_type: LLM
client_prototcol: grpc

genai_perf_flags:
  backend: vllm
  streaming: true
```

For LLMs there are three new metrics being reported: **Inter-token Latency**, **Time to First Token Latency** and **Output Token Throughput**.

These new metrics can be specified as either objectives or constraints.

_**NOTE: In order to enable these new metrics you must enable `streaming` in `genai_perf_flags` and the `client protocol` must be set to `gRPC`**_

---
