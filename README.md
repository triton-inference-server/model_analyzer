[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)](https://opensource.org/licenses/Apache-2.0)

![NVIDIA Logo](../images/NVIDIA-logo.png?raw=true)

# Model Analyzer

Model Analyzer is used for gathering compute and memory requirements of models used with NVIDIA Triton Inference Server. It runs each selected model on Triton Inference Server with the given batch sizes and concurrency values. As the configurations run, Model Analyzer will capture and export the metrics in CSV format to your chosen directory.

These metrics will be listed for each batch size and concurrency value. Baseline metrics for the server running with no models will also be provided. These values can be used for optimizing the loading and running of models.

![Interface](../images/interface-preview.png?raw=true "Model Analyzer Interface")

See [NVIDIA Developer Blog post] for potential use cases.

### Metrics

The metrics collected by Model Analyzer are listed below:

- **Throughput**: Number of inference requests per second
- **Maximum Memory Utilization**: Maximum percentage of time during which global (device) memory was being read or written
- **Maximum GPU Utilization**: Maximum percentage of time during which one or more kernels was executing on the GPU
- **Maximum GPU Memory**: Maximum MB of GPU memory in use


## Documentation

Model Analyzer runs as a command line interface (CLI), standalone Docker container, or Helm chart. Documentation for each is provided in the [docs] folder.

## Quick Start

The [docs] folder hosts all information about building Model Analyzer from source.  
If you want to use Model Analyzer right away, there are two options:

##### You have supported hardware and drivers, [NVIDIA-Docker], and [.NET SDK].

Navigate to the ModelAnalyzer folder, then:


```
git clone https://github.com/NVIDIA/model-analyzer
cd ModelAnalyzer
dotnet publish -c Release
./ModelAnalyzer -m MODELS \
        --model-folder /ABSOLUTE/PATH/TO/MODELS \
        -e /PATH/TO/EXPORT/DIRECTORY \
        --export -b BATCH-SIZES \
        -c CONCURRENCY-VALUES \
        --triton-version TRITON-VERSION \
```

##### You have supported hardware and drivers and [NVIDIA-Docker].

```
git clone https://github.com/NVIDIA/model-analyzer
cd ModelAnalyzer
docker build -f Dockerfile -t model-analyzer .
docker run -v /var/run/docker.sock:/var/run/docker.sock \
        -v /ABSOLUTE/PATH/TO/MODELS:ABSOLUTE/PATH/TO/MODELS \
        -v /ABSOLUTE/PATH/TO/EXPORT/DIRECTORY:/results --net=host \
        model-analyzer:ANALYZER-VERSION \
        --batch BATCH-SIZES --concurrency CONCURRENCY-VALUES \
        --model-names MODEL-NAMES \
        --triton-version TRITON-VERSION \
        --model-folder /ABSOLUTE/PATH/TO/MODELS \
        --export --export-path /results/
```


## Contributing

Contributions to Model Analyzer are more than welcome. To
contribute make a pull request and follow the guidelines outlined in
the [Contributing] document.

## Reporting problems, asking questions

We appreciate any feedback, questions or bug reporting regarding this
project. When help with code is needed, follow the process outlined in
the Stack Overflow (https://stackoverflow.com/help/mcve)
document. Ensure posted examples are:

* minimal – use as little code as possible that still produces the
  same problem

* complete – provide all parts needed to reproduce the problem. Check
  if you can strip external dependency and still show the problem. The
  less time we spend on reproducing problems the more time we have to
  fix it

* verifiable – test the code you're about to provide to make sure it
  reproduces the problem. Remove all other problems that are not
  related to your request/question.
   
## Support

Please start by reviewing the [docs] folder and searching the current project issues. If you do not find a relevant issue, feel free to open one.

[.NET SDK]: https://dotnet.microsoft.com/download
[Contributing]: CONTRIBUTING.md
[docs]: docs
[NVIDIA Developer Blog post]: https://developer.nvidia.com/blog/maximizing-deep-learning-inference-performance-with-nvidia-model-analyzer/
[NVIDIA-Docker]: https://github.com/NVIDIA/nvidia-docker