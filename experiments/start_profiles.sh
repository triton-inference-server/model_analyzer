#!/bin/bash
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CHECKPOINT_DIR="/mnt/nvdl/datasets/inferenceserver/model_analyzer_profile_results"

for model in inception_v1_graphdef resnet50_libtorch vgg19_libtorch; do
  for radius in {2..8}; do
    for magnitude in {1..8}; do
      for min_initialized in {2..8}; do
        echo "Profiling $model (radius = $radius, magnitude = $magnitude, min-initialized = $min_initialized)"
        python3 main.py --save \
            --model-name $model \
            --generator QuickRunConfigGenerator \
            --data-path $CHECKPOINT_DIR \
            --output-path output \
            --radius $radius \
            --magnitude $magnitude \
            --min-initialized $min_initialized \
      done
    done
  done
done
