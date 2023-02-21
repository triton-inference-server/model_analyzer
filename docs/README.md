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

# **Model Analyzer Documentation**

| [Installation](README.md#installation) | [Getting Started](README.md#getting-started) | [User Guide](README.md#user-guide) | [Additional Resources](README.md#resources) |
| -------------------------------------- | -------------------------------------------- | ---------------------------------- | ------------------------------------------- |

## **Installation**

See the [Installation Guide](install.md) for details on how to install Model Analyzer.

## **Getting Started**

- The [Quick Start Guide](quick_start.md) will show you how to use Model Analyzer to profile, analyze and report on a simple PyTorch model.
- Watch the [Optimizing Triton Deployments](https://www.youtube.com/watch?v=UU9Rh00yZMY) video for a step-by-step guide of how the optimal configuration for a BERT model can be found.

## **User Guide**

The User Guide describes how to configure Model Analyzer, choose launch and search modes, and describes what metrics are captured as well as the different reports that can be generated.

- [Model Analyzer CLI](cli.md)
- [Launch Modes](launch_modes.md)
- [Configuring Model Analyzer](config.md)
- [Model Analyzer Metrics](metrics.md)
- [Model Config Search](config_search.md)
- [Checkpointing](checkpoints.md)
- [Model Analyzer Reports](report.md)
- [Deployment with Kubernetes](kubernetes_deploy.md)

## **Resources**

The following resources are recommended:

- [Perf Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md): Perf Analyzer is a CLI application built to generate inference requests and measures the latency of those requests and throughput of the model being served.
