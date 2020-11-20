#!/bin/bash
# Copyright 2020, NVIDIA CORPORATION.
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

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_LINKS_TXT="${DIR}/clara_deploy_pipelines.txt"
MODEL_REPOSITORY_PATH="/qa_model_repository"

rm -rf $MODEL_REPOSITORY_PATH && mkdir -p $MODEL_REPOSITORY_PATH
cd $MODEL_REPOSITORY_PATH

# Get each file in links, unzip and extract model folder
while IFS= read -r line
do
    [[ $line == \#* ]] || [[ -z $line ]] && continue
    wget -q --content-disposition $line -P $MODEL_REPOSITORY_PATH
    unzip files.zip
    unzip app_*model*.zip
    rm *.zip *.yaml *.txt
done < $MODEL_LINKS_TXT
