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
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CHECKPOINT_DIRECTORY="."
TRITON_LAUNCH_MODE="local"
CLIENT_PROTOCOL="http"
PORTS=(`find_available_ports 3`)
MODEL_ANALYZER_GLOBAL_OPTIONS="-v"

# Generate checkpoint by running profile first
echo "Generating checkpoint for add_sub model..."
PROFILE_EXPORT_PATH="${OUTPUT_MODEL_REPOSITORY}/profile_results"
mkdir -p $PROFILE_EXPORT_PATH

MODEL_ANALYZER_PROFILE_ARGS="-m $MODEL_REPOSITORY --profile-models add_sub"
MODEL_ANALYZER_PROFILE_ARGS="$MODEL_ANALYZER_PROFILE_ARGS --triton-launch-mode=$TRITON_LAUNCH_MODE --client-protocol=$CLIENT_PROTOCOL"
MODEL_ANALYZER_PROFILE_ARGS="$MODEL_ANALYZER_PROFILE_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_PROFILE_ARGS="$MODEL_ANALYZER_PROFILE_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_PROFILE_ARGS="$MODEL_ANALYZER_PROFILE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_PROFILE_ARGS="$MODEL_ANALYZER_PROFILE_ARGS -e $PROFILE_EXPORT_PATH --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_PROFILE_ARGS="$MODEL_ANALYZER_PROFILE_ARGS --run-config-search-max-concurrency 2 --run-config-search-max-instance-count 2"

set +e
$MODEL_ANALYZER profile $MODEL_ANALYZER_GLOBAL_OPTIONS $MODEL_ANALYZER_PROFILE_ARGS >> $LOGS_DIR/profile.log 2>&1
PROFILE_RET=$?
set -e

if [ $PROFILE_RET -ne 0 ]; then
    echo -e "\n***\n*** Failed to generate checkpoint. model-analyzer profile exited with non-zero exit code. \n***"
    cat $LOGS_DIR/profile.log
    exit 1
fi

# Find the generated checkpoint and copy it to expected location
GENERATED_CKPT=$(ls -t $CHECKPOINT_DIRECTORY/*.ckpt 2>/dev/null | head -1)
if [ -z "$GENERATED_CKPT" ]; then
    echo -e "\n***\n*** Failed to find generated checkpoint file. \n***"
    ls -la $CHECKPOINT_DIRECTORY
    exit 1
fi
# Only copy if the checkpoint isn't already named 0.ckpt
if [ "$GENERATED_CKPT" != "$CHECKPOINT_DIRECTORY/0.ckpt" ]; then
    cp $GENERATED_CKPT $CHECKPOINT_DIRECTORY/0.ckpt
fi
echo "Checkpoint generated successfully: $GENERATED_CKPT"

MODEL_ANALYZER_ANALYZE_BASE_ARGS="--checkpoint-directory $CHECKPOINT_DIRECTORY --filename-server-only=$FILENAME_SERVER_ONLY"
MODEL_ANALYZER_ANALYZE_BASE_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"
MODEL_ANALYZER_SUBCOMMAND="analyze"
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
