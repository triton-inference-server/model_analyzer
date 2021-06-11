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

# Model Analyzer CLI

Use the `-h` or `--help` flag to view a description of the Model Analyzer's
command line interface.

```
$ model-analyzer -h
```

Options like `-q`, `--quiet` and `-v`, `--verbose` are global and apply to all
model analyzer subcommands.

## Model Analyze Modes

The `-m` or `--mode` flag is global and is accessible to all subcommands. It tells the model analyzer the context
in which it is being run. Currently model analyzer supports 2 modes.

### Online Mode

This is the default mode. When in this mode, Model Analyzer will operate to find
the optimal model configuration for an online inference scenario. In this
scenario, Triton server will receive requests on demand with an expectation that
latency will be minimized.

By default in online mode, the best model configuration will be the one that
minimizes latency. If a latency budget is specified the best model configuration
will be the one with the highest throughput in the given budget. The analyze and
report subcommands also generate summaries specific to online inference. See the
example [online summary](../examples/online_summary.pdf) and [detailed
report](../examples/online_summary.pdf).

### Offline Mode

The offline mode `--mode=offline` tells Model Analyzer to set its defaults to
find a model that maximizes throughput. In the offline scenario, Triton
processes requests offline and therefore inference throughput is the priority. A
minimum throughput can be specified using `--min-throughput` to ignore any
configuration that does not exceed a minimum number of inferences per second.
Both the summary and the detailed report will contain alternative graphs in the
offline mode. See the [offline summary](../examples/offline_summary.pdf) and
[detailed report](../examples/offline_detailed_report.pdf) examples.

## Model Analyzer Subcommands

The Model Analyzer's functionality is split across 3 separate subcommands. Each
subcommand has its own CLI and config options. Some options are required for
more than one subcommand (e.g. `--export-path`). See the [Configuring Model
Analyzer](./config.md) section for more details on configuring each of these
subcommands.

## Subcommand: `profile`

The `profile` subcommand allows the user to run model inferences using perf
analyzer, and collect metrics like throughput, latency and memory usage. Use the
following command to see the usage and argument descriptions for the subcommand.

```
$ model-analyzer profile -h
```

Depending on the command line or YAML config options provided, the `profile`
subcommand will either perform a
[manual](./config_search.md#Manual-Configuration-Search) or [automatic
search](./config_search.md#Automatic-Configuration-Search) over perf analyzer
and model config file parameters. For each combination of [model config
parameters](./config.md#model-config-parameters) (e.g. *instance count* and
*dynamic batch size*), it will run tritonserver and perf analyzer instances with
all the specified run parameters (client request concurrency and static batch
size). It will also save the protobuf (.pbtxt) model config files corresponding
to each combination in the [*output model
repository*](./config.md#CLI-and-YAML-Config-Options). Model Analyzer collects
various metrics at fixed time intervals during these perf analyzer runs. Each
perf analyzer run generates a single measurement, which corresponds to a row in
the output tables. After completing the runs for all configurations for each
model, the Model Analyzer will save the measurements it has collected into the
**checkpoint directory** as a *pickle* file. See the
[Checkpointing](./checkpoints.md) section for more details on checkpoints

### Examples

Some example profile commands are shown here. For a full example see the 
[quick start](./quick_start.md) section.

1. Run auto config search on a model called `resnet50_libtorch` located in `/home/model_repo`

  ```
  $ model-analyzer profile -m /home/model_repo --profile-models resnet50_libtorch
  ```

2. Run auto config search on 2 models called `resnet50_libtorch` and `vgg16_graphdef` located in `/home/model_repo` and save checkpoints to `checkpoints`

  ```
  $ model-analyzer profile -m /home/model_repo --profile-models resnet50_libtorch,vgg16_graphdef --checkpoint-directory=checkpoints
  ```

3.  Run auto config search on a model called `resnet50_libtorch` located in `/home/model_repo`, but change the repository where model config variants are stored to `/home/output_repo`

  ```
  $ model-analyzer profile -m /home/model_repo --output-model-repository-path=/home/output_repo --profile-models resnet50_libtorch
  ```

4. Run profile over manually defined configurations for a models `classification_malaria_v1` and `classification_chestxray_v1` located in `/home/model_repo` using the YAML config file

  ```
  $ model-analyzer profile -f config.yaml
  ```

The contents of `config.yaml` are shown below.
```yaml

model_repository: /home/model_repo

run_config_search_disable: True

concurrency: [2,4,8,16,32]
batch_sizes: [8,16,64]

profile_models: 
  classification_malaria_v1:
    model_config_parameters:
      instance_group:
        -
          kind: KIND_GPU
          count: [1,2]
      dynamic_batching:
        preferred_batch_size: [[32]]
  classification_chestxray_v1:
    model_config_parameters:
      instance_group:
        -
          kind: KIND_GPU
          count: [1,2]
      dynamic_batching:
        preferred_batch_size: [[32]]

```

## Subcommand: `analyze`

The `analyze` subcommand allows the user to create summaries and data tables
from the measurements taken using the `profile` subcommand. The YAML config file
can be used to set constraints and objectives used to sort and filter the
measurements, and order the model configs and models according to the metrics
collected. Use the
following command to see the usage and argument descriptions for the subcommand.

```
$ model-analyzer analyze -h
```

The `analyze` subcommand begins by loading the "latest" checkpoint available in
the checkpoint directory. Next, it sorts the models specified in the CLI or
config YAML, provided they contain measurements in the checkpoint, using the
objectives specified in the config YAML. Finally, it constructs summary PDFs
using the top model configs for each model, as well as across models, if
requested (See the [Reports](./reports.md) section for more details). The
`analyze` subcommand can be run multiple times with different configurations if
the user would like to sort and filter the results using different objectives or
under different constraints.

### Examples

1. Create summary and results for model `resnet50_libtorch` from latest checkpoint in directory `checkpoints`.

  ```
  $ model-analyzer analyze --analysis-models resnet50_libtorch --checkpoint-directory=checkpoints
  ```

2. Create summaries and results for models `resnet50_libtorch` and `vgg16_graphdef` from same checkpoint as above and export them to a directory called `export_directory`

  ```
  $ model-analyzer analyze --analysis-models resnet50_libtorch,vgg16_graphdef -e export_directory --checkpoint-directory=checkpoints
  ```
3. Apply objectives and constraints to sort and filter results in summary plots and tables using yaml config file.

  ```
  $ model-analyzer analyze -f config.yaml
  ```

The contents of `config.yaml` are shown below.

  ```yaml
  checkpoint_directory: ./checkpoints/
  export_path: ./export_directory/

  analysis_models: 
    resnet50_libtorch:
      objectives:
        - perf_throughput
      constraints:
        perf_latency:
          max: 15
    vgg16_graphdef:
      objectives:
        - gpu_used_memory
      constraints:
        perf_latency:
          max: 15
  ```

## Subcommand: `report`

The `report` subcommand allows the user to create detailed reports on one or
more of the model configs that were profiled. 

```
$ model-analyzer report -h
```

Instead of showing only the top measurements from each config like in the
summary reports, Model Analyzer compiles and displays all the meausurements for
a given config in the detailed report (See the [Reports](./reports.md) section
for more details).

### Examples

1. Generate detailed reports for a model configs of `resnet50_libtorch` called `resnet50_libtorch_i1` and `resnet50_libtorch_i2`. Read from `checkpoints` and write to `export_directory`.

  ```
  $ model-analyzer --report-model-configs resnet50_libtorch_i1,resnet50_libtorch_i2 --checkpoint-directory checkpoints -e export_directory
  ```

2. Generate detailed report for `resnet50_libtorch_i2` with a custom plot using YAML config file

  ```
  $ model-analyzer report -f config.yaml
  ```

The contents of the `config.yaml` are shown below

  ```yaml
  checkpoint_directory: ./checkpoints/
  export_path: './export_directory'
  report_model_configs:
    resnet50_libtorch_i2:
      plots:
        throughput_v_memory:
           title: Thoughput vs GPU Memory
           x_axis: gpu_used_memory
           y_axis: perf_throughput
           monotonic: True    
  ```
