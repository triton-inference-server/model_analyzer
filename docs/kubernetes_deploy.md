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

Before deploying the model analyzer, the directories that the container will mount must be specified in `helm-chart/values.yaml`.

```
# Job timeout value specified in seconds
jobTimeout: 900

## Configurations for mounting volumes

# Local path to model directory
modelPath: /home/models

# Local path export model config variants
outputModelPath: /home/output_models

# Local path to export data
resultsPath: /home/results

# Local path to store checkpoints
checkpointPath: /home/checkpoints

## Images
images:

  analyzer:
    image: model-analyzer

  triton:
    image: nvcr.io/nvidia/tritonserver
    tag: 21.07-py3
```

The model analyzer executable uses the config file defined in `helm-chart/templates/config-map.yaml`. This config can be modified to supply arguments to model analyzer. Only the content under the `config.yaml` section of the file should be modified.

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: analyzer-config
  namespace: default
data:
  config.yaml: |
    ######################
    # Config for profile #
    ######################

    override_output_model_repository: True
    run_config_search_disable: True
    triton_http_endpoint: localhost:8000
    triton_grpc_endpoint: localhost:8001
    triton_metrics_url: http://localhost:8002/metrics

    concurrency: 1,2
    batch_sizes: 1

    profile_models: 
      resnet50_libtorch:
        model_config_parameters:
          instance_group:
            -
              kind: KIND_GPU
              count: [1]
          dynamic_batching:
            preferred_batch_size: [[32]]

    ######################
    # Config for analyze #
    ######################
    
    num_configs_per_model: 3

    analysis_models: 
      resnet50_libtorch:
        objectives:
          perf_throughput: 10
        constraints:
          perf_latency:
            max: 15

    ######################
    # Config for report #
    ######################

    report_model_configs:
      - resnet50_libtorch_i0
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

Check that the model analyzer pod is running.

```
~/model_analyzer$ kubectl get pods -A
NAMESPACE     NAME                                             READY   STATUS    RESTARTS   AGE
default       model-analyzer-model-analyzer-t9rsl              1/1     Running   0          23s
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

You can find the results upon completion of the job in the directory passed as the `resultsPath` in `helm-chart/values.yaml`.

```
~/model_analyzer$ ls -l /home/results
total 12
drwxr-xr-x 4 root root 4096 Jun  2 17:00 plots
drwxr-xr-x 4 root root 4096 Jun  2 17:00 reports
drwxr-xr-x 2 root root 4096 Jun  2 17:00 results
```
```
~/model_analyzer$ cat /home/results/results/*
Model,GPU ID,Batch,Concurrency,Model Config Path,Instance Group,Preferred Batch Sizes,Satisfies Constraints,GPU Memory Usage (MB),GPU Utilization (%),GPU Power Usage (W)
resnet50_libtorch,0,1,2,resnet50_libtorch_i0,1/GPU,[32],Yes,1099.0,16.2,85.3
resnet50_libtorch,0,1,1,resnet50_libtorch_i0,1/GPU,[32],Yes,1099.0,14.4,82.2

Model,Batch,Concurrency,Model Config Path,Instance Group,Preferred Batch Sizes,Satisfies Constraints,Throughput (infer/sec),p99 Latency (ms),RAM Usage (MB)
resnet50_libtorch,1,2,resnet50_libtorch_i0,1/GPU,[32],Yes,195.0,11.0,2897.0
resnet50_libtorch,1,1,resnet50_libtorch_i0,1/GPU,[32],Yes,164.0,8.1,2937.0

Model,GPU ID,GPU Memory Usage (MB),GPU Utilization (%),GPU Power Usage (W)
triton-server,0,277.0,0.0,56.5
```
