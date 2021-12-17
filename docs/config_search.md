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

# Model Config Search

Model Analyzer's `profile` subcommand supports **automatic** and **manual**
sweeping through different configurations for Triton models.

## Automatic Configuration Search

Automatic configuration search is the default behavior when running Model
Analyzer. This mode is enabled when there is not any parameter specified for the
`model_config_parameters` section of the Model Analyzer Config. The parameters
that are automatically searched are
[`instance_group`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#instance-groups)
and
[`dynamic_batching`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#dynamic-batcher)
settings.


An example model analyzer config that performs automatic config search looks
like below:

```yaml
model_repository: /path/to/model/repository/

profile_models:
  - model_1
  - model_2
```

In the default mode, automatic config search will try values 1 until 1024 for
[concurrency](https://github.com/triton-inference-server/server/blob/master/docs/perf_analyzer.md#request-concurrency)
increased exponentially (i.e. 1, 2, 4, 8, ...). Maximum value can be configured
using the `run_config_search_max_concurrency` key in the Model Analyzer Config.
For
[`instance_group`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#instance-groups),
Model Analyzer tries values from 1 to 5. This value can be changed using the
`run_config_search_max_instance_count` key in the Model Analyzer Config. For
[`dynamic_batching`](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#dynamic-batcher)
settings, Model Analyzer tries enabling dynamic batching.

An example config that limits the search space used by Model Analyzer is
described below:

```yaml
model_repository: /path/to/model/repository/

run_config_search_max_instance_count: 3
run_config_search_max_concurrency: 8
profile_models:
  - model_1
  - model_2
```

If either `concurrency` or `model_config_parameters` is specified for one of the
models, it will disable the automatic config search for the parameter provided.

For example, the config specified below will only automatically sweep through
the `model_config_parameters` that was described above:

```yaml
model_repository: /path/to/model/repository/

profile_models:
  model_1:
    concurrency: 1,2,3,128
```

The config described below will only sweep through different values for
`concurrency`:

```yaml
model_repository: /path/to/model/repository/

profile_models:
  model_1:
    model_config_parameters:
        instance_group:
        -
            kind: KIND_GPU
            count: [1, 2]
```

If both `concurrency` and `model_config_parameters` are specified, automatic
config search will be disabled.

### Important Note about Remote Mode

In the remote mode, `model_config_parameters` are always ignored because Model
Analyzer has no way of accessing the model repository of the remote Triton
Server. In this mode, only concurrency values can be swept.

## Manual Configuration Search

In addition to the automatic config search, Model Analyzer supports a manual
config search mode. To enable this mode, `--run-config-search-disable` flag
should be provided in the CLI or `run_config_search_disable: True` in the Model
Analyzer Config.

In this mode, values for both `concurrency` and `model_config_parameters` needs
to be specified. If no value for `concurrency` is specified, the default value,
1, will be used. This mode in comparison to the automatic mode, is not limited
to `dynamic_batching` and `instance_count` config parameters. Using manual
config search, you can create custom sweeps for every parameter that can be
specified in the model configuration. Model Analyzer only checks the syntax
of the `model_config_parameters` that is specified and cannot guarantee that
the configuration that is generated is loadable by Triton.

An example Model Analyzer Config that performs manual sweeping looks like below:

```yaml
model_repository: /path/to/model/repository/

run_config_search_disable: True
profile_models:
  model_1:
    model_config_parameters:
        max_batch_size: [6, 8]
        dynamic_batching:
            max_queue_delay_microseconds: [200, 300]
        instance_group:
        -
            kind: KIND_GPU
            count: [1, 2]
```

In this mode, Model Analyzer can sweep through every Triton model configuration
parameter available. For a complete list of parameters allowed under
`model_config_parameters`, refer to the [Triton Model
Configuration](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md).
It is your responsibility to make sure that the sweep configuration specified
works with your model. For example, in the above config, if we change `[6, 8]`
as the range for the `max_batch_size` to `[1]`, it will no longer be a valid
Triton Model Configuration.

The configuration sweep described above, will sweep through 8 configs = (2
`max_batch_size`) * (2 `max_queue_delay_microseconds`) * (2 `instance_group`) values.

### Examples of Additional Model Config Parameters

As mentioned in the previous section, manual configuration search allows you to
sweep on every parameter that can be specified in Triton model configuration. In
this section, we describe some of the parameters that might be of interest for
manual sweep:

* [Rate limiter](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#rate-limiter-config) setting
* If the model is using [ONNX](https://github.com/triton-inference-server/onnxruntime_backend) or [Tensorflow backend](https://github.com/triton-inference-server/tensorflow_backend), the "execution_accelerators" parameters. More information about this parameter is
available in the [Triton Optimization Guide](https://github.com/triton-inference-server/server/blob/main/docs/optimization.md#framework-specific-optimization)
