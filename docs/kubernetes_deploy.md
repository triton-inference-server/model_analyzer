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

# Deploying Model Analyzer on a Kubernetes cluster

Model Analyzer provides support deployment on a Kubernetes enabled
cluster using helm charts. You can find information about helm charts [here](https://helm.sh/).

## Requirements

* A [Kubernetes](https://kubernetes.io/) enabled cluster
* [Helm](https://helm.sh/)

## Using Kubernetes with GPUs

1. **Install Kubernetes :** Follow the steps in the [NVIDIA Kubernetes Installation Docs](https://docs.nvidia.com/datacenter/cloud-native/kubernetes/install-k8s.html) to install Kubernetes, verify your installation, and troubleshoot any issues.

2. **Set Default Container Runtime :** Kubernetes does not yet support the `--gpus` options for running Docker containers, so all GPU nodes will need to register the `nvidia` runtime as the default for Docker on all GPU nodes. Follow the directions in the  [NVIDIA Container Toolkit Installation Docs](https://docs.nvidia.com/datacenter/cloud-native/kubernetes/dcgme2e.html#install-nvidia-container-toolkit-previously-nvidia-docker2).

3. **Install NVIDIA Device Plugin :** The NVIDIA Device Plugin is also required to use GPUs with Kubernetes. The device plugin provides a daemonset that automatically enumerates the number of GPUs on your worker nodes, and allows pods to run on them. Follow the directions in the [NVIDIA Device Plugin Docs](https://docs.nvidia.com/datacenter/cloud-native/kubernetes/dcgme2e.html#install-nvidia-device-plugin) to deploy the device plugin on your cluster.

## Deploy Model Analyzer

To begin, check that your cluster has all the necessary pods deployed.

```
$ kubectl get pods -A
NAMESPACE     NAME                                             READY   STATUS    RESTARTS   AGE
kube-system   calico-kube-controllers-5dc87d545c-5c9sp         1/1     Running   0          21m
kube-system   calico-node-8dcn5                                1/1     Running   0          21m
kube-system   coredns-f9fd979d6-9l29n                          1/1     Running   0          36m
kube-system   coredns-f9fd979d6-mf775                          1/1     Running   0          36m
kube-system   etcd-user.nvidia.com                             1/1     Running   0          36m
kube-system   kube-apiserver-user.nvidia.com                   1/1     Running   0          36m
kube-system   kube-controller-manager-user.nvidia.com          1/1     Running   0          36m
kube-system   kube-proxy-zhpv7                                 1/1     Running   0          36m
kube-system   kube-scheduler-user.nvidia.com                   1/1     Running   0          36m
kube-system   nvidia-device-plugin-1607379880-dblhc            1/1     Running   0          11m
```

Before deploying the model analyzer, you can provide arguments to the model-analyzer executable at `helm-chart/values.yaml`.

```
~/model_analyzer$ cat helm-chart/values.yaml 
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Values for Triton Model Analyzer
# This is a YAML-formatted file.
# Declare variables to be passed into your templates.

# The mandatory fields to update are modelPath, resultsPath, and modelName

# Job timeout value specified in seconds
jobTimeout: 900

## Configurations for mounting volumes

# Local path to model directory
modelPath: /home/models

# Local path to export data
resultsPath: /home/results

## Arguments

#Specifies how long to gather server-only metrics in seconds
durationSeconds: 5

#Specifies list of model batch sizes
batchSizes: 1,2

#Specifies comma-delimited list of concurrency values
concurrency: 1,2

#Specifies frequency of metric gathering in seconds
monitoringInterval: 0.01

#Specifies the max number of any retry attempt in seconds
maxRetries: 100

#Specifies list of model names
#This should match the names Triton is expecting, based on the model's configuration file.
modelNames: classification_chestxray_v1

## Images
images:

  analyzer:
    image: model-analyzer

  triton:
    image: nvcr.io/nvidia/tritonserver
    tag: 21.05-py3
```

Now from the Model Analyzer root directory, we can deploy the helm chart.

```
~/model_analyzer$ helm install model-analyzer helm-chart
NAME: model-analyzer
LAST DEPLOYED: Mon Dec  7 15:09:14 2020
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
```

Check that the model analyzer pod is running and that it has two containers.

```
~/model_analyzer$ kubectl get pods -A
NAMESPACE     NAME                                             READY   STATUS    RESTARTS   AGE
default       model-analyzer-model-analyzer-t9rsl              2/2     Running   0          23s
kube-system   calico-kube-controllers-5dc87d545c-5c9sp         1/1     Running   0          54m
kube-system   calico-node-8dcn5                                1/1     Running   0          54m
kube-system   coredns-f9fd979d6-9l29n                          1/1     Running   0          69m
kube-system   coredns-f9fd979d6-mf775                          1/1     Running   0          69m
kube-system   etcd-user.nvidia.com                             1/1     Running   0          69m
kube-system   kube-apiserver-user.nvidia.com                   1/1     Running   0          69m
kube-system   kube-controller-manager-user.nvidia.com          1/1     Running   0          69m
kube-system   kube-proxy-zhpv7                                 1/1     Running   0          69m
kube-system   kube-scheduler-user.nvidia.com                   1/1     Running   0          69m
kube-system   nvidia-device-plugin-1607379880-dblhc            1/1     Running   0          44m
```

You can find the results upon completion of the job in the directory you passed as the `resultsPath` in `helm-chart/values.yaml`.

```
~/model_analyzer$ cat /home/results/*
Model,Batch,Concurrency,Throughput(infer/sec),Max GPU Utilization(%),Max GPU Used Memory(MB),Max GPU Free Memory(MB)
classification_chestxray_v1,1,1,50.8,38.0,5215.0,23332.0
classification_chestxray_v1,1,2,66.2,40.0,5215.0,19004.0
classification_chestxray_v1,2,1,102.0,39.0,5215.0,19004.0
classification_chestxray_v1,2,2,134.0,42.0,5215.0,19004.0

Model,Batch,Concurrency,Throughput(infer/sec),Max GPU Utilization(%),Max GPU Used Memory(MB),Max GPU Free Memory(MB)
triton-server,0,0,0,0.0,279.0,23940.0
```
