<!--
# Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

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

- [Perf Analyzer](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/perf_analyzer.md): Perf Analyzer is a CLI application built to generate inference requests and measures the latency of those requests and throughput of the model being served.
