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

[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)](https://opensource.org/licenses/Apache-2.0)

# Model Analyzer Helm Chart

## Requirements

In additiom to the base requirements, the Helm version also requires Kubernetes and Helm.

## Procedure

1. Build the Docker container by navigating to the root folder and running `docker build -f Dockerfile -t model-analyzer .'
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
