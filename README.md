<!--
Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.

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

**LATEST RELEASE: You are currently on the main branch which tracks
under-development progress towards the next release. The latest
release of the Triton Model Analyzer is 1.4.0 and is available on
branch
[r21.05](https://github.com/triton-inference-server/model_analyzer/tree/r21.05).**

Triton Model Analyzer is a CLI tool to help with better understanding of the
compute and memory requirements of the Triton Inference Server models. These
reports will help the user better understand the trade-offs in different
configurations and choose a configuration that maximizes the performance of
Triton Inference Server.

## Documentation

* [Installation](docs/install.md)
* [Quick Start](docs/quick_start.md)
* [Model Analyzer CLI](docs/cli.md)
* [Launch Modes](docs/launch_modes.md)
* [Model Analyzer Metrics](docs/metrics.md)
* [Configuring Model Analyzer](docs/config.md)
* [Model Config Search](docs/config_search.md)
* [Checkpointing](docs/checkpoints.md)
* [Model Analyzer Reports](docs/report.md)
* [Deployment with Kubernetes](docs/kubernetes_deploy.md)

# Reporting problems, asking questions

We appreciate any feedback, questions or bug reporting regarding this
project. When help with code is needed, follow the process outlined in
the Stack Overflow (https://stackoverflow.com/help/mcve)
document. Ensure posted examples are:

* minimal – use as little code as possible that still produces the
  same problem

* complete – provide all parts needed to reproduce the problem. Check
  if you can strip external dependency and still show the problem. The
  less time we spend on reproducing problems the more time we have to
  fix it

* verifiable – test the code you're about to provide to make sure it
  reproduces the problem. Remove all other problems that are not
  related to your request/question.
