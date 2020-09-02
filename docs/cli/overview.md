# Model Analyzer Command Line Interface (CLI)

## Overview

Model Analyzer is responsible for gathering compute metrics on a model. It runs a selected model on a Triton Inference Server with the given batch sizes and concurrency values. As the configurations run, Model Analyzer will capture and export the metrics in CSV format to your chosen directory.

These metrics will be listed for each batch size and concurrency value. Baseline metrics for the server running with no models will also be provided. These values can be used for optimizing the loading and running of models.

## Requirements

Model Analyzer supports the following products and environments:
- All K80 and newer Tesla GPUs
- NVSwitch on DGX-2 and HGX-2
- All Maxwell and newer non-Tesla GPUs
- CUDA 7.5+ and NVIDIA Driver R384+

Model Analyzer requires NVIDIA-Docker, a supported Tesla Recommend Driver, and a supported CUDA toolkit. The CLI also requires a .NET Core SDK.

Ports 8000, 8001, and 8002 on your network need to be available for the Triton server. Model Analyzer supports Triton Inference Server version 20.02-py3. If you do not have the Triton server image locally, the first run will take a couple of minutes to pull it.

## Procedure

1. Compile the CLI by navigating to the `src` folder and running `dotnet publish -c Release'
2. Then, change directories into the folder where Model Analyzer was published: `cd bin/Release/netcoreapp3.1/linux-x64/publish/`
3. You can type `./model-analyzer` to see command line argument options. You can also run the below command, replacing the capitalized arguments:

        ./model-analyzer -m MODELS \
        --model-folder /ABSOLUTE/PATH/TO/MODELS \
        -e /PATH/TO/EXPORT/DIRECTORY \
        --export -b BATCH-SIZES \
        -c CONCURRENCY-VALUES \
        --triton-version TRITON-VERSION \

Please reference the below sample command:

        ./model-analyzer -m chest_xray,covid19_xray \
        --model-folder /home/user/models \
        -e /home/user/results \
        --export -b 1,4,8 \
        -c 2,4,8 \
        --triton-version 20.02-py3

4. When the CLI finishes running, you will see the results outputted to the screen. If you enabled the `--export` flag, navigate to your specified export directory to find the exported results in CSV format.

## Metrics

The metrics collected by Model Analyzer are listed below:

- **Throughput**: Number of inference requests per second
- **Maximum Memory Utilization**: Maximum percentage of time during which global (device) memory was being read or written
- **Maximum GPU Utilization**: Maximum percentage of time during which one or more kernels was executing on the GPU
- **Maximum GPU Memory**: Maximum MB of GPU memory in use