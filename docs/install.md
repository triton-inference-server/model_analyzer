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

# Install 

There are three ways to use Triton Model Analyzer:

1. Using the `pip`:
   ```
   $ pip3 install git+https://github.com/triton-inference-server/model_analyzer
   ```

   If you are using this approach you need to install [tritonclient](https://github.com/triton-inference-server/server/blob/master/docs/client_libraries.md) and DCGM on your
   machine.

   For installing DCGM on Ubuntu 20.04 you can use the following commands:
   ```
   $ export DCGM_VERSION=2.0.13
   $ wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    dpkg -i datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb
   ```

2. Building from source:

   Building from source is similar to installing using pip. You need
   to install the same dependencies mentioned in the "Using pip section".
   After that, you can use the following commands:

   ```
   $ git clone https://github.com/triton-inference-server/model_analyzer
   $ cd model_analyzer
   $ pip3 install setup.py
   ```

   After these steps, `model-analyzer` executable should be available in
   `$PATH`.

3. Building the Dockerfile:
   ```
   $ docker build --pull -t modelanalyzer .
   $ docker run -ti -v <load triton models> --gpus 1 -v `pwd`/examples/quick-start:/workspace/examples modelanalyzer bash
   ```