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

# Ensemble Model Quick Start

The steps below will guide you through using Model Analyzer in Docker mode to profile and analyze a simple ensemble model: ensemble_add_sub.

## `Step 1:` Download the ensemble model `ensemble_add_sub` and composing models `add`, `sub`

---

**1. Create a new directory and enter it**

```
mkdir <new_dir> && cd <new_dir>
```

**2. Start a git repository**

```
git init && git remote add -f origin https://github.com/triton-inference-server/model_analyzer.git
```

**3. Enable sparse checkout, and download the examples directory, which contains the ensemble_add_sub, add and sub**

```
git config core.sparseCheckout true && \
echo 'examples' >> .git/info/sparse-checkout && \
git pull origin main
```

**3. Add a version directory to ensemble_add_sub**

```
mkdir examples/quick/ensemble_add_sub/1
```

## `Step 2:` Pull and Run the SDK Container

---

**1. Pull the SDK container:**

```
docker pull nvcr.io/nvidia/tritonserver:23.07-py3-sdk
```

**2. Run the SDK container**

```
docker run -it --gpus 1 \
      --shm-size 1G \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v $(pwd)/examples/quick-start:$(pwd)/examples/quick-start \
      -v <path-to-output-model-repo>:<path-to-output-model-repo> \
      --net=host nvcr.io/nvidia/tritonserver:23.07-py3-sdk
```

**Replacing** `<path-to-output-model-repo>` with the
**_absolute_ _path_** to the directory where the output model repository
will be located.
This ensures the Triton SDK container has access to the model
config variants that Model Analyzer creates.<br><br>
**Important:** You must ensure the absolutes paths are identical on both sides of the mounts (or else Tritonserver cannot load the model)<br><br>
**Important:** The example above uses a single GPU. If you are running on multiple GPUs, you need to increase the shared memory size accordingly<br><br>

## `Step 3:` Profile the `ensemble_add_sub` model

---

The [examples/quick-start](../examples/quick-start) directory is an example
[Triton Model Repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md) that contains the ensemble model `ensemble_add_sub`, which calculates the sum and difference of two inputs using `add` and `sub` models. 

Run the Model Analyzer `profile` subcommand inside the container with:

```
model-analyzer profile \
    --model-repository <path-to-examples-quick-start> \
    --profile-models ensemble_add_sub \
    --triton-launch-mode=docker --triton-docker-shm-size=1G \
    --output-model-repository-path <path-to-output-model-repo>/<output_dir> \
    --export-path profile_results
```

**Important:** You must specify an `<output_dir>` subdirectory. You cannot have `--output-model-repository-path` point directly to `<path-to-output-model-repo>`

**Important:** If you already ran this earlier in the container, you can use the `--override-output-model-repository` option to overwrite the earlier results.

**Important**: All models must be in the same repository

---

The `--run-config-profile-models-concurrently-enable` option tells Model Analyzer to load and optimize both models concurrently using the [Quick Search](config_search.md#quick-search-mode) algorithm.

This will profile both models concurrently finding the maximal throughput gain for both models by iterating across instance group counts and batch sizes. By default, the algorithm is attempting to find the best balance of gains for each model, not the best combined total throughput.

After the quick search completes, Model Analyzer will then sweep concurrencies for the top three (default) configurations and then create a summary report and CSV outputs. We can specify the top-N configurations by using `--num-configs-per-model`.

---

Here is an example result summary, run on a Tesla V100 GPU:

![Result Summary Top](../examples/ensemble_result_summary_top.jpg)
![Result Summary Table](../examples/ensemble_result_summary_table.jpg)

You will note that the top model configuration has a higher throughput than the other configurations.

---

The measured data and summary report will be placed inside the
`./profile_results` directory. The directory will be structured as follows.

```
$HOME
|-- model_analyzer
    |-- profile_results
        |-- plots
        |   |-- detailed
        |   |   |-- ensemble_add_sub_config_5
        |   |   |   `-- latency_breakdown.png
        |   |   |-- ensemble_add_sub_config_6
        |   |   |   `-- latency_breakdown.png
        |   |   `-- ensemble_add_sub_config_7
        |   |       `-- latency_breakdown.png
        |   `-- simple
        |       |-- ensemble_add_sub
        |       |   |-- gpu_mem_v_latency.png
        |       |   `-- throughput_v_latency.png
        |       |-- ensemble_add_sub_config_5
        |       |   |-- cpu_mem_v_latency.png
        |       |   |-- gpu_mem_v_latency.png
        |       |   |-- gpu_power_v_latency.png
        |       |   `-- gpu_util_v_latency.png
        |       |-- ensemble_add_sub_config_6
        |       |   |-- cpu_mem_v_latency.png
        |       |   |-- gpu_mem_v_latency.png
        |       |   |-- gpu_power_v_latency.png
        |       |   `-- gpu_util_v_latency.png
        |       `-- ensemble_add_sub_config_7
        |           |-- cpu_mem_v_latency.png
        |           |-- gpu_mem_v_latency.png
        |           |-- gpu_power_v_latency.png
        |           `-- gpu_util_v_latency.png
        |-- reports
        |   |-- detailed
        |   |   |-- ensemble_add_sub_config_5
        |   |   |   `-- detailed_report.pdf
        |   |   |-- ensemble_add_sub_config_6
        |   |   |   `-- detailed_report.pdf
        |   |   `-- ensemble_add_sub_config_7
        |   |       `-- detailed_report.pdf
        |   `-- summaries
        |       `-- ensemble_add_sub
        |           `-- result_summary.pdf
        `-- results
            |-- metrics-model-gpu.csv
            |-- metrics-model-inference.csv
            `-- metrics-server-only.csv
```