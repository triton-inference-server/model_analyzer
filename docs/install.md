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

# Installation

There are three ways to use Triton Model Analyzer:

1. Building the Dockerfile:

   The recommended way to use Model Analyzer is with [`docker`](https://docs.docker.com/get-docker/). First, clone the Model Analyzer's git repository,
   then build the docker image.

   ```
   $ git clone https://github.com/triton-inference-server/model_analyzer
   $ docker build --pull -t model-analyzer .
   ```

   The above command will pull all the containers that model analyzer needs to run. The Model Analyzer's Dockerfile bases the container on the latest `tritonserver` containers from NGC. Now you can run the container with:

   ```
   $ docker run -it --privileged --rm --gpus all \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v <model_repo_abs_path>:<model_repo_abs_path> \
        -v <workspace_path>:/opt/triton-model-analyzer \
        -v <output_repo_abs_path>:<output_repo_abs_path> \
        --net=host --name model-analyzer 
   
   root@hostname:/opt/triton-model-analyzer# 
   ```

2. Using `pip3`:

   You can install pip using:
   ```
   $ sudo apt-get update && sudo apt-get install python3-pip
   ```
   
   Model analyzer can be installed with: 
   ```
   $ pip3 install nvidia-pyindex
   $ pip3 install triton-model-analyzer
   ```

   If you encounter any errors installing dependencies like `numba`, make sure that you have the latest version of `pip` using:

   ```
   $ pip3 install --upgrade pip
   ```
   
   You can then try installing model analyzer again.

   If you are using this approach you need to install [tritonclient](https://github.com/triton-inference-server/server/blob/master/docs/client_libraries.md) and DCGM on your
   machine.

   For installing DCGM on Ubuntu 20.04 you can use the following commands:
   ```
   $ export DCGM_VERSION=2.0.13
   $ wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    dpkg -i datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb
   ```

3. Building from source:

   You may sometimes want to build from source, if for example you want to use a 
   custom version of `perf_analyzer`. You'll need to install the same dependencies (tritonclient and DCGM) mentioned in the "Using pip section". After that, you can use the following commands:

   ```
   $ git clone https://github.com/triton-inference-server/model_analyzer
   $ cd model_analyzer
   $ ./build_wheel.sh <path to perf_analyzer> true
   ```

   In the final command above we are building the triton-model-analyzer wheel. You will need to provide the `build_wheel.sh` script with two arguments. The first is the path to the `perf_analyzer` binary that you would like Model Analyzer to use. The second is whether you want this wheel to be linux specific. Currently, this argument must be set to `true` as perf analyzer is supported only on linux. This will create a wheel file in the `wheels` directory named `triton-model-analyzer-<version>-py3-none-manylinux1_x86_64.whl`. We can now install this with:

   ```
   $ pip3 install wheels/triton-model-analyzer-*.whl
   ```

   After these steps, `model-analyzer` executable should be available in
   `$PATH`.
