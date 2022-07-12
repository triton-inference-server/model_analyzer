#!/bin/bash

CHECKPOINT_DIR="/mnt/nvdl/datasets/inferenceserver/model_analyzer_profile_results"

for model in inception_v1_graphdef resnet50_libtorch vgg19_libtorch; do
  for radius in {2..8}; do
    for magnitude in {1..8}; do
      for min_initialized in {2..8}; do
        echo "Profiling $model (radius = $radius, magnitude = $magnitude, min-initialized = $min_initialized)"
        python3 main.py --save \
            --model-name $model \
            --generator UndirectedRunConfigGenerator \
            --data-path $CHECKPOINT_DIR \
            --output-path output \
            --radius $radius \
            --magnitude $magnitude \
            --min-initialized $min_initialized \
      done
    done
  done
done
