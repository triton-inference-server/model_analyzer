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

# BLS Model Quick Start

The steps below will guide you through using Model Analyzer in Docker mode to profile and analyze a simple BLS model: bls.

## `Step 1:` Download the BLS model `bls` and composing model `add`

---

**1. Create a new directory and enter it**

```
mkdir <new_dir> && cd <new_dir>
```

**2. Start a git repository**

```
git init && git remote add -f origin https://github.com/triton-inference-server/model_analyzer.git
```

**3. Enable sparse checkout, and download the examples directory, which contains the bls and add models**

```
git config core.sparseCheckout true && \
echo 'examples' >> .git/info/sparse-checkout && \
git pull origin main
```

## `Step 2:` Pull and Run the SDK Container

---

**1. Pull the SDK container:**

```
docker pull nvcr.io/nvidia/tritonserver:24.07-py3-sdk
```

**2. Run the SDK container**

```
docker run -it --gpus 1 \
      --shm-size 2G \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v $(pwd)/examples/quick-start:$(pwd)/examples/quick-start \
      --net=host nvcr.io/nvidia/tritonserver:24.07-py3-sdk
```

**Important:** The example above uses a single GPU. If you are running on multiple GPUs, you may need to increase the shared memory size accordingly<br><br>

## `Step 3:` Profile the `bls` model

---

The [examples/quick-start](../examples/quick-start) directory is an example [Triton Model Repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md) that contains the BLS model `bls` which calculates the sum of two inputs using `add` model.

An example model analyzer YAML config that performs a BLS model search

```
model_repository: <path-to-examples-quick-start>
profile_models:
  - bls
bls_composing_models: add
perf_analyzer_flags:
  input-data: <path-to-examples-bls_input_data.json>
triton_launch_mode: docker
triton_docker_shm_size: 2G
output_model_repository_path: <path-to-output-model-repo>/<output_dir>
export_path: profile_results
```

**Important:** You must specify an `<output_dir>` subdirectory. You cannot have `output_model_repository_path` point directly to `<path-to-output-model-repo>`

**Important:** If you already ran this earlier in the container, you can overwrite earlier results by adding the `override_output_model_repository: true` field to the YAML file.

**Important**: All models must be in the same repository

**Important:** [`bls`](../examples/quick-start/bls) model takes "MODEL_NAME" as one of its inputs. We must include "add" in the input data JSON file as "MODEL_NAME" for this example to function. Otherwise, Perf Analyzer will produce random data for "MODEL_NAME," resulting in failed inferences.

Run the Model Analyzer `profile` subcommand inside the container with:

```
model-analyzer profile -f /path/to/config.yml
```

---

The Model analyzer uses [Quick Search](config_search.md#quick-search-mode) algorithm for profiling the BLS model. After the quick search is completed, Model Analyzer will then sweep concurrencies for the top three configurations and then create a summary report and CSV outputs.

Here is an example result summary, run on a Tesla V100 GPU:

![Result Summary Top](../examples/bls_result_summary_top.jpg)
![Result Summary Table](../examples/bls_result_summary_table.jpg)

You will note that the top model configuration has a higher throughput than the other configurations.

---

The measured data and summary report will be placed inside the
`./profile_results` directory. The directory will be structured as follows.

```
$HOME
|-- model_analyzer
    |-- profile_results
        |-- perf_analyzer_error.log
        |-- plots
        |   |-- detailed
        |   |   |-- bls_config_7
        |   |   |   `-- latency_breakdown.png
        |   |   |-- bls_config_8
        |   |   |   `-- latency_breakdown.png
        |   |   `-- bls_config_9
        |   |       `-- latency_breakdown.png
        |   `-- simple
        |       |-- bls
        |       |   |-- gpu_mem_v_latency.png
        |       |   `-- throughput_v_latency.png
        |       |-- bls_config_7
        |       |   |-- cpu_mem_v_latency.png
        |       |   |-- gpu_mem_v_latency.png
        |       |   |-- gpu_power_v_latency.png
        |       |   `-- gpu_util_v_latency.png
        |       |-- bls_config_8
        |       |   |-- cpu_mem_v_latency.png
        |       |   |-- gpu_mem_v_latency.png
        |       |   |-- gpu_power_v_latency.png
        |       |   `-- gpu_util_v_latency.png
        |       `-- bls_config_9
        |           |-- cpu_mem_v_latency.png
        |           |-- gpu_mem_v_latency.png
        |           |-- gpu_power_v_latency.png
        |           `-- gpu_util_v_latency.png
        |-- reports
        |   |-- detailed
        |   |   |-- bls_config_7
        |   |   |   `-- detailed_report.pdf
        |   |   |-- bls_config_8
        |   |   |   `-- detailed_report.pdf
        |   |   `-- bls_config_9
        |   |       `-- detailed_report.pdf
        |   `-- summaries
        |       `-- bls
        |           `-- result_summary.pdf
        `-- results
            |-- metrics-model-gpu.csv
            |-- metrics-model-inference.csv
            `-- metrics-server-only.csv
```

**Note:** Above configurations, bls_config_7, bls_config_8, and bls_config_9 are generated as the top configurations when running profiling on a single Tesla V100 GPU. However, running on multiple GPUs or different model GPUs may result in different top configurations.
