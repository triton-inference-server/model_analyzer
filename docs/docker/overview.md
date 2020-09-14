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

# Triton Memory Analyzer Docker Container

## Procedure

1. Build the Docker container by navigating to the root folder and running `docker build -f Dockerfile -t memory-analyzer .'
2. Run the below command, replacing the capitalized arguments:

        docker run -v /var/run/docker.sock:/var/run/docker.sock \
        -v /ABSOLUTE/PATH/TO/MODELS:/ABSOLUTE/PATH/TO/MODELS \
        -v /ABSOLUTE/PATH/TO/EXPORT/DIRECTORY:/results --net=host \
        memory-analyzer:ANALYZER-VERSION \
        --batch BATCH-SIZES --concurrency CONCURRENCY-VALUES \
        --model-names MODEL-NAMES \
        --triton-version TRITON-VERSION \
        --model-folder /ABSOLUTE/PATH/TO/MODELS \
        --export --export-path /results/

Please reference the below sample command:

        docker run -v /var/run/docker.sock:/var/run/docker.sock \
        -v /home/user1/models:/home/user1/models \
        -v /home/user1/results:/results --net=host \
        memory-analyzer:0.7.3-2008.1 \
        --batch 1,4,8 --concurrency 2,4,8 \
        --model-names chest_xray,covid19_xray\
        --triton-version 20.02-py3 \
        --model-folder /home/user1/models \
        --export --export-path /results/

3. When the container completes running, you will see the results output to the screen. If you enabled the `--export` flag, navigate to your specified export directory to find the exported results in CSV format.
