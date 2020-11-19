<!--
Copyright 2020, NVIDIA CORPORATION.
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

# Quick Start

The goal of this section is to analyze a simple libtorch model.

1. Build Model Anlayzer Container:

```
docker build . -t triton_modelanalyzer
```

2. Running Triton Model Analyzer Container:

The [examples/quick-start](../examples/quick-start) directory contains a simple libtorch model calculates
the sum and difference of two inputs.

```
$ docker run --gpus 1 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -ti -v `pwd`/examples/quick-start:/workspace/examples triton_modelanalyzer bash
```

3. Run the Model Analyzer inside the container:

```
model-analyzer -m /workspace/examples/ -n add_sub
```

You should see an output similar to the output below:

```
*** Measurement Settings *** 
  Batch size: 1
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 14601
    Throughput: 2920.2 infer/sec
    Avg latency: 341 usec (standard deviation 103 usec)
    p50 latency: 280 usec
    p90 latency: 503 usec
    p95 latency: 513 usec
    p99 latency: 633 usec
    Avg HTTP time: 325 usec (send/recv 34 usec + response wait 291 usec)
  Server:
    Inference count: 18180
    Execution count: 18180
    Successful request count: 18180
    Avg request latency: 143 usec (overhead 20 usec + queue 13 usec + compute input 28 usec + compute infer 46 usec + compute output 36 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 2920.2 infer/sec, latency 341 usec

Server Only: 
Model           Batch   Concurrency   Throughput(infer/sec)   Max GPU Utilization(%)   Max GPU Used Memory(MB)   Max GPU Free Memory(MB)
triton-server   0       0             0                       0.0                      700.0                     23519.0

Models:
Model           Batch   Concurrency   Throughput(infer/sec)   Max GPU Utilization(%)   Max GPU Used Memory(MB)   Max GPU Free Memory(MB)
add_sub         1       1             2920.2                  3.0                      702.0                     23519.0
```

Note that the exact numbers may be different depending on the GPU your using.
