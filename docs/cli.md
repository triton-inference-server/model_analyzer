<!--
Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# Model Analyzer CLI

Use the `-h` or `--help` flag to view a description of the Model Analyzer's
command line interface.

```
$ model-analyzer -h
```

## Model Analyzer Modes

The `-m` or `--mode` flag is global and is accessible to all subcommands. It tells the model analyzer the context
in which it is being run. Currently model analyzer supports 2 modes.

### Online Mode

This is the default mode. When in this mode, Model Analyzer will operate to find
the optimal model configuration for an online inference scenario. By default in
online mode, the best model configuration will be the one that maximizes
throughput. If a latency budget is specified to the [profile subcommand](#subcommand-profile) via
`--latency-budget`, then the best model configuration will be the one with the highest throughput in the given budget.

In online mode the profile and report subcommands will generate summaries specific to online inference.
See the example [online summary](../examples/online_summary.pdf) and [online detailed report](../examples/online_detailed_report.pdf).

### Offline Mode

The offline mode `--mode=offline` tells Model Analyzer to operate to find the
optimal model configuration for an offline inference scenario. By default
in offline mode, the best model configuration will be the one that maximizes throughput.
A minimum throughput can be specified to the [profile subcommand](#subcommand-profile)
via `--min-throughput` to ignore any configuration that does not exceed a minimum number of inferences per second.

In offline mode the profile and report subcommands will generate reports specific to offline inference.
See the example [offline summary](../examples/offline_summary.pdf) and
[offline detailed report](../examples/offline_detailed_report.pdf) examples.

## Model Analyzer Subcommands

The Model Analyzer's functionality is split across three separate subcommands. Each
subcommand has its own CLI and config options. Some options are required for
more than one subcommand (e.g. `--export-path`). See the [Configuring Model
Analyzer](./config.md) section for more details on configuring each of these
subcommands.

## Subcommand: `profile`

The `profile` subcommand begins by loading the "latest" checkpoint (if available) in
the checkpoint directory. It will then run model inferences using perf
analyzer, and collect metrics like throughput, latency and memory usage for
any measurements not present in the checkpoint.

Next, it sorts the models specified in the CLI or
config YAML, using any objectives specified in the config YAML. Finally, it constructs summary PDFs
using the top model configs for each model, as well as across models, if
requested (See the [Reports](./report.md) section for more details).

The `profile` subcommand can be run multiple times with different configurations if
the user would like to sort and filter the results using different objectives or
under different constraints.

Use the following command to see the usage and argument descriptions for the subcommand.

```
$ model-analyzer profile -h
```

Depending on the command line or YAML config options provided, the `profile`
subcommand will either perform a
[manual](./config_search.md#manual-brute-search), [automatic](./config_search.md#automatic-brute-search), or
[quick](./config_search.md#quick-search-mode) search over perf analyzer
and model config file parameters. For each combination of [model config
parameters](./config.md#model-config-parameters) (e.g. _max batch size_, _dynamic batching_, and _instance group count_), it will run tritonserver and perf analyzer instances with
all the specified run parameters (client request concurrency and static batch
size). It will also save the protobuf (.pbtxt) model config files corresponding
to each combination in the [_output model
repository_](./config.md#cli-and-yaml-config-options). Model Analyzer collects
various metrics at fixed time intervals during these perf analyzer runs. Each
perf analyzer run generates a single measurement, which corresponds to a row in
the output tables. After completing the runs for all configurations for each
model, the Model Analyzer will save the measurements it has collected into the
**checkpoint directory**. See the
[Checkpointing](./checkpoints.md) section for more details on checkpoints

### Examples

Some example profile commands are shown here. For a full example see the
[quick start](./quick_start.md) section.

**Note: All commands assume that you are running in directory where MA was installed**

1. Run auto config search on a model called `add_sub` located in `examples/quick-start`

```
$ model-analyzer profile -m examples/quick-start --profile-models add_sub
```

2. Run quick search on a model called `add_sub` located in `examples/quick-start`

```
$ model-analyzer profile -m examples/quick-start --profile-models add_sub --run-config-search-mode quick
```

3. Run auto config search on 2 models called `add` and `sub` located in `examples/quick-start` and save checkpoints to `checkpoints`

```
$ model-analyzer profile -m examples/quick-start --profile-models add,sub --checkpoint-directory=checkpoints
```

4.  Run auto config search on a model called `add_sub` located in `examples/quick-start`, but change the repository where model config variants are stored to `/home/output_repo`

```
$ model-analyzer profile -m examples/quick-start --output-model-repository-path=/home/output_repo --profile-models add_sub
```

5. Run profile over manually defined configurations for a models `add` and `sub` located in `examples/quick-start` using the YAML config file

```
$ model-analyzer profile -f /path/to/config.yaml
```

The contents of `config.yaml` are shown below.

```yaml
model_repository: examples/quick-start

run_config_search_disable: True

concurrency: [2, 4, 8, 16, 32]
batch_sizes: [8, 16, 64]

profile_models:
  add:
    model_config_parameters:
      instance_group:
        - kind: KIND_GPU
          count: [1, 2]
      dynamic_batching:
        max_queue_delay_microseconds: [100]
  sub:
    model_config_parameters:
      instance_group:
        - kind: KIND_GPU
          count: [1, 2]
      dynamic_batching:
        max_queue_delay_microseconds: [100]
```

6. Apply objectives and constraints to sort and filter results in summary plots and tables using yaml config file.

```
$ model-analyzer profile -f /path/to/config.yaml
```

The contents of `config.yaml` are shown below.

```yaml
checkpoint_directory: ./checkpoints/
export_path: ./export_directory/

profile_models:
  add:
    objectives:
      - perf_throughput
    constraints:
      perf_latency_p99:
        max: 15
  sub:
    objectives:
      - gpu_used_memory
    constraints:
      perf_latency_p99:
        max: 15
```

**Note**: The checkpoint directory should be removed between consecutive runs of
the `model-analyzer profile` command when you do not want to include the results
from a previous profile.

## Subcommand: `report`

The `report` subcommand allows the user to create detailed reports on one or
more of the model configs that were profiled.

```
$ model-analyzer report -h
```

Instead of showing only the top measurements from each config like in the
summary reports, Model Analyzer compiles and displays all the meausurements for
a given config in the detailed report (See the [Reports](./report.md) section
for more details).

### Examples

1. Generate detailed reports for a model configs of `add_sub` called `add_sub_config_1` and `add_sub_config_2`. Read from `checkpoints` and write to `export_directory`.

```
$ model-analyzer --report-model-configs add_sub_config_1,add_sub_config_2 --checkpoint-directory checkpoints -e export_directory
```

2. Generate detailed report for `add_sub_config_2` with a custom plot using YAML config file

```
$ model-analyzer report -f /path/to/config.yaml
```

The contents of the `config.yaml` are shown below

```yaml
checkpoint_directory: ./checkpoints/
export_path: "./export_directory"
report_model_configs:
  add_sub_config_2:
    plots:
      throughput_v_memory:
        title: Thoughput vs GPU Memory
        x_axis: gpu_used_memory
        y_axis: perf_throughput
        monotonic: True
```
