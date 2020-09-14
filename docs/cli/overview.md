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

# Triton Memory Analyzer Command Line Interface (CLI)

## Requirements

In addition to the base requirements, the CLI requires .NET Core SDK.

## Procedure

1. Compile the CLI by navigating to the `src` folder and running `dotnet publish -c Release'
2. Then, change directories into the folder where Triton Memory Analyzer was published: `cd bin/Release/netcoreapp3.1/linux-x64/publish/`
3. You can type `./memory-analyzer` to see command line argument options. You can also run the below command, replacing the capitalized arguments:

        ./memory-analyzer -m MODELS \
        --model-folder /ABSOLUTE/PATH/TO/MODELS \
        -e /PATH/TO/EXPORT/DIRECTORY \
        --export -b BATCH-SIZES \
        -c CONCURRENCY-VALUES \
        --triton-version TRITON-VERSION \

Please reference the below sample command:

        ./memory-analyzer -m chest_xray,covid19_xray \
        --model-folder /home/user/models \
        -e /home/user/results \
        --export -b 1,4,8 \
        -c 2,4,8 \
        --triton-version 20.02-py3

4. When the CLI finishes running, you will see the results outputted to the screen. If you enabled the `--export` flag, navigate to your specified export directory to find the exported results in CSV format.
