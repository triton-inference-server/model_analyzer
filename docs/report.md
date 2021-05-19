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

# Model Analyzer Reports

The Model analyzer can run inferences with models across multiple parameters. It
can also generate reports containing useful plots such as the throughput versus
latency curve obtained from these inference runs, as well as tables describing
the best performing model configurations for each model.

## Summary Reports

The most basic type of report is the *Summary Report* which the Model Analyzer's
`analyze` subcommand generates by default for each model.

```
$ model-analyzer analyze --analysis-models <list of model names> --checkpoint-directory <path to checkpoints directory> -e <path to export directory> -f <optional config file>
```

The export directory will, by default, contain 3 subdirectories. The summary
report for a model will be located in `[export-path]/reports/summaries/<model
name>`. The report will look like the one shown [*here*](../examples/summary.pdf).

To disable summary report generation use `--summarize=false` or set the
`summarize` yaml option to `false`.


## Detailed Reports

The second type of report is the *Detailed Report* which can be generated using
the Model Analyzer's `report` subcommand. 

```
$ model-analyzer report --report-model-configs <list of model configs> --checkpoint-directory <path to checkpoints directory> -e <path to export directory> -f <optional config file>
```

You will be able to locate the detailed report for a model config under
`[export-path]/reports/detailed/<model config name>` The detailed reports
contain a detailed breakdown plot of the throughput vs latency for the
particular model config with which the measurements are obtained, as well as
extra configurable plots. The user can define the plots they would like to see
in the detailed report using the YAML config file (See [**Configuring Model
Analyzer**](./config.md) section for more details) The detailed report will
look like the one shown [*here*](../examples/detailed_report.pdf).


See the [**quick start**](./quick_start.md#plots) and [**configuring model
analyzer**](./config.md) sections for more details on how to configure these
reports.
