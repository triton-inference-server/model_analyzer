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
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/nvdl/datasets/inferenceserver/$REPO_VERSION/libtorch_model_store"}
CHECKPOINT_REPOSITORY=${CHECKPOINT_REPOSITORY:="/mnt/nvdl/datasets/inferenceserver/model_analyzer_checkpoints/2022_08_02"}
QA_MODELS="resnet50_libtorch"
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

cp $CHECKPOINT_REPOSITORY/resnet50_vgg19.ckpt $CHECKPOINT_DIRECTORY/0.ckpt
rm -rf $OUTPUT_MODEL_REPOSITORY


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
