<!--
Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

Triton Model Analyzer's `profile` subcommand supports four different launch
modes along with Triton Inference Server. In the `local` and `docker` modes,
Triton Inference Server will be launched by the Model Analyzer. In the `c_api`
mode, the Triton Inference Server is launched locally via a C API. In the
`remote` mode, it is assumed there is an already running instance of Triton
Inference Server.

### Docker

| CLI Option | **`--triton-launch-mode=docker`** |
| - | - |

Note: A full step by step example of docker mode can be found in the [Quick Start Guide](quick_start.md).

In this mode, Model Analyzer uses the Python Docker API to launch the Triton
Inference Server container. If you are running Model Analyzer inside a Docker
container, make sure that the container is launched with appropriate flags. The
following flags are mandatory for correct behavior:

```
--gpus <gpus> -v /var/run/docker.sock:/var/run/docker.sock --net host
```

Additionally, Model Analyzer uses the `output_model_repository_path` to
manipulate and store model config variants. When Model Analyzer launches the
Triton container, it does so as a *sibling container*. The launched Triton
container will only have access to the host filesystem. **As a result, in the
docker launch mode, the output model directory will need to be mounted to the
Model Analyzer docker container at the same absolute path it has outside the
container.** So you must add the following when you launch the model analyzer
container as well.

```
-v <path-to-output-model-repository>:<path-to-output-model-repository>
```

Finally, when launching model analyzer, the argument `--output-model-repository`
must be provided as a directory inside `<path-to-output-model-repository>`. This
directory need not exist.

```
--output-model-repository=<path-to-output-model-repository>/output
```

This mode is useful if you want to use the Model Analyzer installed in the
Triton SDK Container. You will need Docker installed, though.

### Local

| CLI Option | **`--triton-launch-mode=local`** |
| - | - |

Local mode is the default mode if no `triton-launch-mode` is specified.

In this mode, Model Analyzer will launch Triton Server using the local binary
supplied using `--triton-server-path`, or if none is supplied, the
`tritonserver` binary in `$PATH`.

There are multiple ways to get Model Analyzer and TritonServer executables together for local mode,
such as [building a container](install.md#specific-version-with-local-launch-mode) that contains both, or [pip installing](install.md#pip) Model analyzer wherever you already
have a TritonServer executable

### C API

| CLI Option | **`--triton-launch-mode=c_api`** |
| - | - |

In this mode, Triton server is launched locally via the
[C_API](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inference_protocols.md#in-process-triton-server-api)
by the perf_analyzer instances launched by Model Analyzer. Inference requests are
sent directly via C-API function calls instead of going through the network via
GRPC or HTTP.

This mode is useful if you want to run with the Triton Server installed locally
and want the increased performance from the C API. Similar to the
[local mode](#local), Triton Server must be installed in the environment that
the Model Analyzer is being used.

The server metrics that Model Analyzer gathers and reports are not available directly
from the triton server when running in C-API mode. Instead, Model Analyzer will attempt to
gather this information itself. This can lead to less precise results, and will generally result
in GPU utilization and power numbers being underreported.

### Remote

| CLI Option | **`--triton-launch-mode=remote`** |
| - | - |

This mode is beneficial when you want to use an already running Triton Inference
Server. You may provide the URLs for the Triton instance's HTTP or GRPC endpoint
depending on your chosen client protocol using the `--triton-grpc-endpoint`, and
`--triton-http-endpoint` flags. You should also make sure that same GPUs are
available to the Inference Server and Model Analyzer and they are on the same
machine. Model Analyzer does not currently support profiling remote GPUs. Triton
Server in this mode needs to be launched with `--model-control-mode explicit`
flag to support loading/unloading of the models. The model parameters cannot be
changed in remote mode, though.
