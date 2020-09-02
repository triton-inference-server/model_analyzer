# Model Analyzer Docker Container

## Overview

Model Analyzer is responsible for gathering compute metrics on a model. It runs a selected model on a Triton Inference Server with the given batch sizes and concurrency values. As the configurations run, Model Analyzer will capture and export the metrics in CSV format to your chosen directory.

These metrics will be listed for each batch size and concurrency value. Baseline metrics for the server running with no models will also be provided. These values can be used for optimizing the loading and running of models.

## Requirements

Model Analyzer supports the following products and environments:

- All K80 and newer Tesla GPUs
- NVSwitch on DGX-2 and HGX-2
- All Maxwell and newer non-Tesla GPUs
- CUDA 7.5+ and NVIDIA Driver R384+

Model Analyzer requires a supported Tesla Recommend Driver and a supported CUDA toolkit.

Ports 8000, 8001, and 8002 on your network need to be available for the Triton server. Model Analyzer supports Triton Inference Server version 20.02-py3. If you do not have the Triton server image locally, the first run will take a couple of minutes to pull it.

## Procedure

1. Build the Docker container by navigating to the root folder and running `docker build -f Dockerfile -t model-analyzer .'
2. Run the below command, replacing the capitalized arguments:

        docker run -v /var/run/docker.sock:/var/run/docker.sock \
        -v /ABSOLUTE/PATH/TO/MODELS:ABSOLUTE/PATH/TO/MODELS \
        -v /ABSOLUTE/PATH/TO/EXPORT/DIRECTORY:/results --net=host \
        model-analyzer:ANALYZER-VERSION \
        --batch BATCH-SIZES --concurrency CONCURRENCY-VALUES \
        --model-names MODEL-NAMES \
        --triton-version TRITON-VERSION \
        --model-folder /ABSOLUTE/PATH/TO/MODELS \
        --export --export-path /results/

Please reference the below sample command:

        docker run -v /var/run/docker.sock:/var/run/docker.sock \
        -v /home/user/models:/home/user/models \
        -v /home/user/results:/results --net=host \
        model-analyzer:0.7.3-2008.1 \
        --batch 1,4,8 --concurrency 2,4,8 \
        --model-names chest_xray,covid19_xray\
        --triton-version 20.02-py3 \
        --model-folder /home/user/models \
        --export --export-path /results/

3. When the container completes running, you will see the results output to the screen. If you enabled the `--export` flag, navigate to your specified export directory to find the exported results in CSV format.

## Metrics

The metrics collected by Model Analyzer are listed below:

- **Throughput**: Number of inference requests per second
- **Maximum Memory Utilization**: Maximum percentage of time during which global (device) memory was being read or written
- **Maximum GPU Utilization**: Maximum percentage of time during which one or more kernels was executing on the GPU
- **Maximum GPU Memory**: Maximum MB of GPU memory in use