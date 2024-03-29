#!/bin/bash
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

source ../common/util.sh
source ../common/check_analyzer_results.sh
create_logs_dir "L0_output_fields"

python3 config_generator.py

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/opt/triton-model-analyzer/examples/quick-start"}
CHECKPOINT_REPOSITORY=${CHECKPOINT_REPOSITORY:="/mnt/nvdl/datasets/inferenceserver/model_analyzer_checkpoints/2023_09_07"}
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CHECKPOINT_DIRECTORY="."

cp $CHECKPOINT_REPOSITORY/add_sub.ckpt $CHECKPOINT_DIRECTORY/0.ckpt

MODEL_ANALYZER_ANALYZE_BASE_ARGS="--checkpoint-directory $CHECKPOINT_DIRECTORY --filename-server-only=$FILENAME_SERVER_ONLY"
MODEL_ANALYZER_ANALYZE_BASE_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"
MODEL_ANALYZER_SUBCOMMAND="analyze"
MODEL_ANALYZER_GLOBAL_OPTIONS="-v"
LIST_OF_CONFIG_FILES=(`ls | grep .yml`)

RET=0

if [ ${#LIST_OF_CONFIG_FILES[@]} -le 0 ]; then
    echo -e "\n***\n*** Test Failed. No config file exists. \n***"
    RET=1
    exit $RET
fi

# Run the analyzer with various configurations and check the results
RET=0

for CONFIG_FILE in ${LIST_OF_CONFIG_FILES[@]}; do

    TEST_NAME=test_$(basename "$CONFIG_FILE" | sed 's/\.[^.]*$//')
    create_result_paths -test-name $TEST_NAME -checkpoints false

    set +e

    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS -e $EXPORT_PATH -f $CONFIG_FILE"

    TEST_OUTPUT_NUM_ROWS=47
    run_analyzer
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
        cat $ANALYZER_LOG
        RET=1
    else
        SERVER_METRICS_FILE=${EXPORT_PATH}/results/${FILENAME_SERVER_ONLY}
        MODEL_METRICS_GPU_FILE=${EXPORT_PATH}/results/${FILENAME_GPU_MODEL}
        MODEL_METRICS_INFERENCE_FILE=${EXPORT_PATH}/results/${FILENAME_INFERENCE_MODEL}

        METRICS_NUM_COLUMNS_FILE=`echo $CONFIG_FILE | sed 's/\.yml//'`-param-gpu.txt
        SERVER_NUM_COLUMNS_FILE=`echo $CONFIG_FILE | sed 's/\.yml//'`-param-server.txt
        INFERENCE_NUM_COLUMNS_FILE=`echo $CONFIG_FILE | sed 's/\.yml//'`-param-inference.txt
        METRICS_NUM_COLUMNS=`cat $METRICS_NUM_COLUMNS_FILE`
        INFERENCE_NUM_COLUMNS=`cat $INFERENCE_NUM_COLUMNS_FILE`
        SERVER_METRICS_NUM_COLUMNS=`cat $SERVER_NUM_COLUMNS_FILE`

        check_table_row_column \
            $ANALYZER_LOG $ANALYZER_LOG $ANALYZER_LOG \
            $MODEL_METRICS_INFERENCE_FILE $MODEL_METRICS_GPU_FILE $SERVER_METRICS_FILE \
            $INFERENCE_NUM_COLUMNS $TEST_OUTPUT_NUM_ROWS \
            $METRICS_NUM_COLUMNS $TEST_OUTPUT_NUM_ROWS \
            $SERVER_METRICS_NUM_COLUMNS 1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed.\n***"
            cat $ANALYZER_LOG
            RET=1
        fi
    fi

    set -e
done

rm -f *.ckpt
rm -rf *.txt
rm -rf *.yml

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
