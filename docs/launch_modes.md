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
# Launch Modes

Triton Model Analyzer's `profile` subcommand supports three different launch
modes along with Triton Server. In the `local` and `docker` modes, Triton
Inference Server will be launched by the Model Analyzer. In the `remote` mode,
it is assumed there is an already running instance of Triton Inference Server.

1. **`tritonserver` binary is available in `$PATH`**. In this case you can use
   the `--triton-launch-mode local` flag. Model Analyzer will launch a Triton
   Inference Server that will be used for benchmarking the models.

2. **Using Docker API to launch Triton Inference Server container.** If you are
   using this mode and you are using Model Analzyer inside a Docker container,
   make sure that the container is launched with appropriate flags. The
   following flags are mandatory for correct behavior:
   ```
   --gpus 1 -v /var/run/docker.sock:/var/run/docker.sock --net host
   ```

   You should use `--triton-launch-mode docker` flag for the Model Analyzer to
   use this mode.

3. **Using an already running Triton Inference Server**. This mode is beneficial
   when you want to use an already running Triton Inference Server. You should
   use `--triton-launch-mode remote` flag to use this mode. You may provide the
   URLs for the Triton instance's HTTP or GRPC endpoint depending on your chosen
   client protocol using the `--triton-grpc-endpoint`, and
   `--triton-http-endpoint` flags.  You should also make sure that same GPUs are
   available to the Inference Server and Model Analyzer and they are on the same
   machine. Model Analyzer does not currently support profiling remote GPUs. You
   should also start the Triton Server using `--model-control-mode=explicit` to
   use this mode.
