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
create_logs_dir "L0_results"

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/opt/triton-model-analyzer/examples/quick-start"}
QA_MODELS="add_sub"
MODEL_NAMES="$(echo $QA_MODELS | sed 's/ /,/g')"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
TRITON_LAUNCH_MODE="local"
CLIENT_PROTOCOL="http"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
create_result_paths

rm -rf $OUTPUT_MODEL_REPOSITORY

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
$MODEL_ANALYZER profile -v $MODEL_ANALYZER_PROFILE_ARGS >> $TEST_LOG_DIR/profile.log 2>&1
PROFILE_RET=$?
set -e

if [ $PROFILE_RET -ne 0 ]; then
    echo -e "\n***\n*** Failed to generate checkpoint. model-analyzer profile exited with non-zero exit code. \n***"
    cat $TEST_LOG_DIR/profile.log
    exit 1
fi

# Find the generated checkpoint and copy it to expected location
GENERATED_CKPT=$(ls -t $CHECKPOINT_DIRECTORY/*.ckpt 2>/dev/null | head -1)
if [ -z "$GENERATED_CKPT" ]; then
    echo -e "\n***\n*** Failed to find generated checkpoint file. \n***"
    ls -la $CHECKPOINT_DIRECTORY
    exit 1
fi
cp $GENERATED_CKPT $CHECKPOINT_DIRECTORY/0.ckpt
echo "Checkpoint generated successfully: $GENERATED_CKPT"


MODEL_ANALYZER_ANALYZE_BASE_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS -e $EXPORT_PATH --checkpoint-directory $CHECKPOINT_DIRECTORY --filename-server-only=$FILENAME_SERVER_ONLY"
MODEL_ANALYZER_ANALYZE_BASE_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"

python3 test_config_generator.py -m $MODEL_NAMES

# Run the analyzer and check the results
RET=0

set +e
CONFIG_FILE='config-summaries.yml'
TEST_NAME='summaries'
ANALYZER_LOG="$TEST_LOG_DIR/analyzer.test_$TEST_NAME.log"

MODEL_ANALYZER_SUBCOMMAND="analyze"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS -f $CONFIG_FILE"
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -d $EXPORT_PATH
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $ANALYZER_LOG.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi
fi

CONFIG_FILE='config-detailed-reports.yml'
TEST_NAME='detailed_reports'
ANALYZER_LOG="$TEST_LOG_DIR/analyzer.test_$TEST_NAME.log"

MODEL_ANALYZER_SUBCOMMAND="report"
MODEL_ANALYZER_ARGS="-e $EXPORT_PATH -f $CONFIG_FILE --checkpoint-directory $CHECKPOINT_DIRECTORY"
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -d $EXPORT_PATH
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $ANALYZER_LOG.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
