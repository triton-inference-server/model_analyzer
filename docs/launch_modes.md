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

1. **`--triton-launch-mode=local`** In this mode, Model Analyzer will launch
   Triton Server using the local binary supplied using `--triton-server-path`,
   or if none is supplied, the `tritonserver` binary in `$PATH`.

2. **`--triton-launch-mode=docker`** In this mode, Model Analyzer uses the
   python Docker API to launch the Triton Inference Server container. If you are
   running Model Analyzer inside a Docker container, make sure that the
   container is launched with appropriate flags. The following flags are
   mandatory for correct behavior:
   ```
   --gpus <gpus> -v /var/run/docker.sock:/var/run/docker.sock --net host
   ```
   
3. **`--triton-launch-mode=C_API`** In this mode, Triton server is launched
   locally via the C_API by the perf_analyzer instances launched by Model
   Analyzer.

4. **`--triton-launch-mode=remote`**. This mode is beneficial when you want to
   use an already running Triton Inference Server. You may provide the URLs for
   the Triton instance's HTTP or GRPC endpoint depending on your chosen client
   protocol using the `--triton-grpc-endpoint`, and `--triton-http-endpoint`
   flags.  You should also make sure that same GPUs are available to the
   Inference Server and Model Analyzer and they are on the same machine. Model
   Analyzer does not currently support profiling remote GPUs.
