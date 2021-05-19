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

# Model Analyzer Metrics

Model Analyzer collects a variety of metrics. Shown below is a list of the
metrics that can be collected using the Model Analyzer, as well as their metric
tags, which are used in various places to configure Model Analyzer.

## Perf Analyzer Metrics

These metrics come from the perf analyzer and are parsed and processed by the
model analyzer. See the [perf analyzer
docs](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md)
for more info on these 

* `perf_throughput`: The number of inferences per second measured by the perf
  analyzer.
* `perf_latency`: The p99 latency as measured by perf analyzer.
* `perf_client_response_wait`: The time spent waiting for a response from the
  server, after an inference request has been sent.
* `perf_client_send_recv`: The total amount of time it takes the client to send
  a request, plus the amount of time it takes for the client to receive the
  response. (Not including network RTT).
* `perf_server_queue`: The average time spent in the inference schedule queue by
  a request waiting for an instance of the model to become available.
* `perf_server_compute_input`: Time needed to copy data to the GPU from input
  buffers
* `perf_server_compute_infer`: The average time spent performing the actual
  inference.
* `perf_server_compute_output`: Time needed to copy data from the GPU to output
  buffers.

## GPU metrics

These are metrics currently captured using DCGM. They are recorded for each GPU
in fixed intervals during perf analyzer runs and then aggregated across all the
records for a run. 

* `gpu_used_memory`: The maximum memory used by the GPU
* `gpu_free_memory`: The maximum memory available in the GPU
* `gpu_utilization`: The average utilization of the GPU
* `gpu_power_usage`: The average power usage of the GPU

## CPU metrics

These metrics are captured using `psutil` or `docker stats`, and are also
recorded and aggregated over fixed intervals during a perf analyzer run.

* `cpu_used_ram`: The total amount of memory used by all CPUs
* `cpu_available_ram`: The total amount of availble CPU memory.

## Additional tags for output headers

These tags are used in options like `server_output_fields`,
`inference_output_fields`, and `gpu_output_fields` to control parameters (not
just metrics) that should be displayed in the output tables.

* `model_name`: Name of the model
* `batch_size`: Batch size used for measurement
* `concurrency`: Client request conccurency used for measurement
* `model_config_path`: The path to the model config
* `instance_group`: The number/type of instances
* `dynamic_batch_sizes`: The values passed as preferred batch sizes to the
  dynamic batcher
* `satisfies_constraints`: `Yes` if this measurement satisfies constraints, `No`
  otherwise.
* `gpu_id`: The id of the GPU this measurement was taken on.
