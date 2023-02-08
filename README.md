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

Triton Model Analyzer is a CLI tool to help with better understanding of the
compute and memory requirements of the
[Triton Inference Server](https://github.com/triton-inference-server/server/) models. These
reports will help the user better understand the trade-offs in different
configurations and choose a configuration that maximizes the performance of
Triton Inference Server.

## Features

- [Brute and Quick search](docs/config_search.md): Model Analyzer can
  help you automatically find the optimal settings for
  [Max Batch Size](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#maximum-batch-size),
  [Dynamic Batching](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher), and
  [Instance Group](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
  parameters of your model configuration. Model Analyzer utilizes
  [Performance Analyzer](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/perf_analyzer.md)
  to test the model with different concurrency and batch sizes of requests. Using
  [Manual Config Search](docs/config_search.md#manual-brute-search), you can create manual sweeps for every parameter that can be specified in the model configuration.

- [Multi-Model Search](docs/config_search.md#multi-model-search-mode): **EARLY ACCESS** - Model Analyzer can help you
  find the optimal settings when profiling multiple concurrent models, utilizing the Quick Search alogrithm

- [Ensemble Model Search](docs/config_search.md#ensemble-model-search): Model Analyzer can help you find the optimal
  settings when profiling a non-BLS ensemble model, utilizing the Quick Search algorithm

- [Detailed and summary reports](docs/report.md): Model Analyzer is able to generate
  summarized and detailed reports that can help you better understand the trade-offs
  between different model configurations that can be used for your model.

- [QoS Constraints](docs/config.md#constraint): Constraints can help you
  filter out the Model Analyzer results based on your QoS requirements. For
  example, you can specify a latency budget to filter out model configurations
  that do not satisfy the specified latency threshold.

## Examples and Tutorials

See the [Quick Start](docs/quick_start.md) for a guide of how to use Model Analyzer to profile, analyze and report on a simple PyTorch model.

## Documentation

- [Installation](docs/install.md)
- [Model Analyzer CLI](docs/cli.md)
- [Launch Modes](docs/launch_modes.md)
- [Configuring Model Analyzer](docs/config.md)
- [Model Analyzer Metrics](docs/metrics.md)
- [Model Config Search](docs/config_search.md)
- [Checkpointing](docs/checkpoints.md)
- [Model Analyzer Reports](docs/report.md)
- [Deployment with Kubernetes](docs/kubernetes_deploy.md)

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
