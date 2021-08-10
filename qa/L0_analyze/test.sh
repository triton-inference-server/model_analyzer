# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

rm -f *.log

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
CHECKPOINT_REPOSITORY=${CHECKPOINT_REPOSITORY:="/mnt/dldata/inferenceserver/model_analyzer_checkpoints"}
QA_MODELS="vgg19_libtorch resnet50_libtorch"
MODEL_NAMES="$(echo $QA_MODELS | sed 's/ /,/g')"
EXPORT_PATH="`pwd`/results"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
CHECKPOINT_DIRECTORY="./checkpoints"

# Create results/checkpoints directories
mkdir $EXPORT_PATH
mkdir $CHECKPOINT_DIRECTORY && cp $CHECKPOINT_REPOSITORY/analyze.ckpt $CHECKPOINT_DIRECTORY/0.ckpt

# Run the analyzer and check the results
RET=0

set +e

ANALYZER_LOG="test_table.log"
MODEL_ANALYZER_ARGS="--analysis-models $MODEL_NAMES --checkpoint-directory $CHECKPOINT_DIRECTORY -e $EXPORT_PATH --filename-server-only=$FILENAME_SERVER_ONLY"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"
MODEL_ANALYZER_SUBCOMMAND="analyze"
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    SERVER_METRICS_FILE=${EXPORT_PATH}/results/${FILENAME_SERVER_ONLY}
    MODEL_METRICS_GPU_FILE=${EXPORT_PATH}/results/${FILENAME_GPU_MODEL}
    MODEL_METRICS_INFERENCE_FILE=${EXPORT_PATH}/results/${FILENAME_INFERENCE_MODEL}
    METRICS_NUM_COLUMNS=11
    METRICS_NUM_ROWS=8
    INFERENCE_NUM_COLUMNS=9
    SERVER_NUM_COLUMNS=5
    SERVER_NUM_ROWS=1

    check_table_row_column \
        $ANALYZER_LOG $ANALYZER_LOG $ANALYZER_LOG \
        $MODEL_METRICS_INFERENCE_FILE $MODEL_METRICS_GPU_FILE $SERVER_METRICS_FILE \
        $INFERENCE_NUM_COLUMNS $METRICS_NUM_ROWS \
        $METRICS_NUM_COLUMNS $METRICS_NUM_ROWS \
        $SERVER_NUM_COLUMNS $SERVER_NUM_ROWS
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi
fi
set -e

# Next also check for muted logs
ANALYZER_LOG="test_quiet.log"
MODEL_ANALYZER_GLOBAL_OPTIONS="-q"
MODEL_ANALYZER_SUBCOMMAND="analyze"
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    # Check that the table is not present
    if [[ ! -z `grep 'Models (Inference)' ${ANALYZER_LOG}` ]]; then
        echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND did not mute table output. \n***"
        cat $ANALYZER_LOG
        RET=1
    fi
fi

rm -rf $EXPORT_PATH/*
rm -f $CHECKPOINT_DIRECTORY/*.ckpt

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
