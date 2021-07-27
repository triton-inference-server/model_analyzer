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

1. The recommended way to use Model Analyzer is with the Triton SDK docker
   container available on the [NVIDIA GPU Cloud
   Catalog](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver). You
   can pull and run the SDK container with the following commands:

   ```
   $ docker pull nvcr.io/nvidia/tritonserver:21.07-py3-sdk
   ```

   If you are not planning to run Model Analyzer with
   `--triton-launch-mode=docker` you can run the container with the following
   command:

   ```
   $ docker run -it --gpus all --net=host nvcr.io/nvidia/tritonserver:21.07-py3-sdk
   ```

   If you intend to use `--triton-launch-mode=docker`, you will need to mount
   the following: 
      * `-v /var/run/docker.sock:/var/run/docker.sock` allows running docker
        containers as sibling containers from inside the Triton SDK container.
        Model Analyzer will require this if run  with
        `--triton-launch-mode=docker`.
      * `-v <path-to-output-model-repo>:<path-to-output-model-repo>` The
        ***absolute*** path to the directory where the output model repository
        will be located (i.e. parent directory of the output model repository).
        This is so that the launched Triton container has access to the model
        config variants that Model Analyzer creates.

   ```
   $ docker run -it --gpus all \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v <path-to-output-model-repo>:<path-to-output-model-repo> \
        --net=host nvcr.io/nvidia/tritonserver:21.07-py3-sdk
   ```

   Model Analyzer uses `pdfkit` for report generation. If you are running Model
   Analyzer inside the Triton SDK container, then you will need to download
   `wkhtmltopdf`.

   ```
   $ sudo apt-get update && sudo apt-get install wkhtmltopdf
   ```

   Once you do this, Model Analyzer will able to use `pdfkit` to generate
   reports.

2. Building the Dockerfile:

   You can also build the Model Analyzer's dockerfile yourself. First, clone the
   Model Analyzer's git repository, then build the docker image.

   ```
   $ git clone https://github.com/triton-inference-server/model_analyzer
   $ docker build --pull -t model-analyzer .
   ```

   The above command will pull all the containers that model analyzer needs to
   run. The Model Analyzer's Dockerfile bases the container on the latest
   `tritonserver` containers from NGC. Now you can run the container with:

   ```
   $ docker run -it --rm --gpus all \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v <path-to-triton-model-repository>:<path-to-triton-model-repository> \
        -v <path-to-output-model-repo>:<path-to-output-model-repo> \
        --net=host model-analyzer

   root@hostname:/opt/triton-model-analyzer# 
   ```

3. Using `pip3`:

   You can install pip using:
   ```
   $ sudo apt-get update && sudo apt-get install python3-pip
   ```

   Model analyzer can be installed with: 
   ```
   $ pip3 install nvidia-pyindex
   $ pip3 install triton-model-analyzer
   ```

   If you encounter any errors installing dependencies like `numba`, make sure
   that you have the latest version of `pip` using:

   ```
   $ pip3 install --upgrade pip
   ```

   You can then try installing model analyzer again.

   If you are using this approach you need to install DCGM on your machine.

   For installing DCGM on Ubuntu 20.04 you can use the following commands:
   ```
   $ export DCGM_VERSION=2.0.13
   $ wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    dpkg -i datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb
   ```

4. Building from source:

   To build model analyzer form source, you'll need to install the same
   dependencies (tritonclient and DCGM) mentioned in the "Using pip section".
   After that, you can use the following commands:

   ```
   $ git clone https://github.com/triton-inference-server/model_analyzer
   $ cd model_analyzer
   $ ./build_wheel.sh <path to perf_analyzer> true
   ```

   In the final command above we are building the triton-model-analyzer wheel.
   You will need to provide the `build_wheel.sh` script with two arguments. The
   first is the path to the `perf_analyzer` binary that you would like Model
   Analyzer to use. The second is whether you want this wheel to be linux
   specific. Currently, this argument must be set to `true` as perf analyzer is
   supported only on linux. This will create a wheel file in the `wheels`
   directory named
   `triton-model-analyzer-<version>-py3-none-manylinux1_x86_64.whl`. We can now
   install this with:

   ```
   $ pip3 install wheels/triton-model-analyzer-*.whl
   ```

   After these steps, `model-analyzer` executable should be available in
   `$PATH`.

**Notes:**
* Triton Model Analyzer supports all the GPUs supported by the DCGM library. See
  [DCGM Supported
  GPUs](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-user-guide/getting-started.html#supported-platforms)
  for more information.

