<!--
Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)](https://opensource.org/licenses/Apache-2.0)

# Triton Model Analyzer

> [!IMPORTANT]
> Model Analyzer support is deprecated and excluded from Triton Inference Server starting in 25.05

> [!Warning]
>
> ##### LATEST RELEASE
>
> You are currently on the `main` branch which tracks under-development progress towards the next release. <br>
> The latest release of the Triton Model Analyzer is 1.47.0 and is available on branch
> [r24.12](https://github.com/triton-inference-server/model_analyzer/tree/r24.12).

Triton Model Analyzer is a CLI tool which can help you find a more optimal configuration, on a given piece of hardware, for single, multiple, ensemble, or BLS models running on a [Triton Inference Server](https://github.com/triton-inference-server/server/). Model Analyzer will also generate reports to help you better understand the trade-offs of the different configurations along with their compute and memory requirements.
<br><br>

# Features

### Search Modes

- [Optuna Search](docs/config_search.md#optuna-search-mode) **_-ALPHA RELEASE-_** allows you to search for every parameter that can be specified in the model configuration, using a hyperparameter optimization framework. Please see the [Optuna](https://optuna.org/) website if you are interested in specific details on how the algorithm functions.

- [Quick Search](docs/config_search.md#quick-search-mode) will **sparsely** search the [Max Batch Size](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#maximum-batch-size),
  [Dynamic Batching](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher), and
  [Instance Group](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups) spaces by utilizing a heuristic hill-climbing algorithm to help you quickly find a more optimal configuration

- [Automatic Brute Search](docs/config_search.md#automatic-brute-search) will **exhaustively** search the
  [Max Batch Size](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#maximum-batch-size),
  [Dynamic Batching](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher), and
  [Instance Group](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
  parameters of your model configuration

- [Manual Brute Search](docs/config_search.md#manual-brute-search) allows you to create manual sweeps for every parameter that can be specified in the model configuration

### Model Types

- [Ensemble](docs/model_types.md#ensemble): Model Analyzer can help you find the optimal
  settings when profiling an ensemble model

- [BLS](docs/model_types.md#bls): Model Analyzer can help you find the optimal
  settings when profiling a BLS model

- [Multi-Model](docs/model_types.md#multi-model): Model Analyzer can help you
  find the optimal settings when profiling multiple concurrent models

- [LLM](docs/model_types.md#llm): Model Analyzer can help you
  find the optimal settings when profiling Large Language Models

### Other Features

- [Detailed and summary reports](docs/report.md): Model Analyzer is able to generate
  summarized and detailed reports that can help you better understand the trade-offs
  between different model configurations that can be used for your model.

- [QoS Constraints](docs/config.md#constraint): Constraints can help you
  filter out the Model Analyzer results based on your QoS requirements. For
  example, you can specify a latency budget to filter out model configurations
  that do not satisfy the specified latency threshold.
  <br><br>

# Examples and Tutorials

### **Single Model**

See the [Single Model Quick Start](docs/quick_start.md) for a guide on how to use Model Analyzer to profile, analyze and report on a simple PyTorch model.

### **Multi Model**

See the [Multi-model Quick Start](docs/mm_quick_start.md) for a guide on how to use Model Analyzer to profile, analyze and report on two models running concurrently on the same GPU.

### **Ensemble Model**

See the [Ensemble Model Quick Start](docs/ensemble_quick_start.md) for a guide on how to use Model Analyzer to profile, analyze and report on a simple Ensemble model.

### **BLS Model**

See the [BLS Model Quick Start](docs/bls_quick_start.md) for a guide on how to use Model Analyzer to profile, analyze and report on a simple BLS model.
<br><br>

# Documentation

- [Installation](docs/install.md)
- [Model Analyzer CLI](docs/cli.md)
- [Launch Modes](docs/launch_modes.md)
- [Configuring Model Analyzer](docs/config.md)
- [Model Analyzer Metrics](docs/metrics.md)
- [Model Config Search](docs/config_search.md)
- [Model Types](docs/model_types.md)
- [Checkpointing](docs/checkpoints.md)
- [Model Analyzer Reports](docs/report.md)
- [Deployment with Kubernetes](docs/kubernetes_deploy.md)
  <br><br>

# Terminology

Below are definitions of some commonly used terms in Model Analyzer:

- **Model Type** - Category of model being profiled. Examples of this include single, multi, ensemble, BLS, etc..
- **Search Mode** - How Model Analyzer explores the possible configuration space when profiling. This is either exhaustive (brute) or heuristic (quick/optuna).
- **Model Config Search** - The cross product of model type and search mode.
- **Launch Mode** - How the Triton Server is deployed and used by Model Analyzer.

# Reporting problems, asking questions

We appreciate any feedback, questions or bug reporting regarding this
project. When help with code is needed, follow the process outlined in
the Stack Overflow (https://stackoverflow.com/help/mcve)
document. Ensure posted examples are:

- minimal – use as little code as possible that still produces the
  same problem

- complete – provide all parts needed to reproduce the problem. Check
  if you can strip external dependency and still show the problem. The
  less time we spend on reproducing problems the more time we have to
  fix it

- verifiable – test the code you're about to provide to make sure it
  reproduces the problem. Remove all other problems that are not
  related to your request/question.
