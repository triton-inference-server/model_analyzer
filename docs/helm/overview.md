# Model Analyzer Helm Chart

## Overview

The Model Analyzer is responsible for gathering compute metrics on a model.

Model Analyzer runs the selected model on Triton Inference Server with the given batch sizes and concurrency values. As the configurations run, Model Analyzer will capture and export the metrics in CSV format to your chosen directory.

These metrics will be listed for each batch size and concurrency value. Baseline metrics for the server running with no models will also be provided. These values can be used for optimizing the loading and running of models.

Model Analyzer supports Triton Inference Server version 20.02-py3. Please note that this version of Model Analyzer supports analyzing one model per job. If you would like to import a list of models, please use the standalone Docker container or command line interface.

## Requirements

Model Analyzer supports the following products and environments:
- All K80 and newer Tesla GPUs
- NVSwitch on DGX-2 and HGX-2
- All Maxwell and newer non-Tesla GPUs
- CUDA 7.5+ and NVIDIA Driver R384+

Model Analyzer requires a supported Tesla Recommend Driver, a supported CUDA toolkit, and NVIDIA-Docker. The Helm version also requires Kubernetes and Helm.

## Procedure

1. Build the Docker container by navigating to the `src` folder and running `docker build -f Dockerfile -t model-analyzer .'
2. Change directories into `helm-chart` folder.
3. Create the below text file as `config.yaml`, updating the field values:

        batch: BATCH_SIZES
        concurrency: CONCURRENCY_VALUES
        modelName: MODEL_NAME
        modelPath: PATH_TO_MODEL_FOLDER
        resultsPath: PATH_TO_RESULTS_FOLDER

A sample `config.yaml` file is below:

        batch: 1,4,8
        concurrency: 2,4,8
        modelName: segmentation_chest_xray
        modelPath: /home/user/models
        resultsPath: /home/user/results

4. Start the Helm chart: `helm install -f config.yaml . `
5. You can watch the Kubernetes pod with `watch kubectl get pods`. The pod will continue running until it times out or is manually stopped. The metrics are ready when there is 1 out of 2 containers running. For most models, this takes 1-2 minutes with one batch size and one concurrency value.
6. You can view the exported metrics, which will be in the directory specified in the values file.

If the pod status is not "pending" or "running", you can use the command `kubectl describe pod <pod-name>` to see more details about the pod.

## Metrics

The metrics collected by Model Analyzer are listed below:

- **Throughput**: Number of inference requests per second
- **Maximum Memory Utilization**: Maximum percentage of time during which global (device) memory was being read or written
- **Maximum GPU Utilization**: Maximum percentage of time during which one or more kernels was executing on the GPU
- **Maximum GPU Memory**: Maximum MB of GPU memory in use
