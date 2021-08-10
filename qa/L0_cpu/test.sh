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

ANALYZER_LOG="cpu.log"
source ../common/util.sh
source ../common/check_analyzer_results.sh

rm -rf results && mkdir -p results
rm -rf checkpoints && mkdir checkpoints
rm -f $ANALYZER_LOG

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/data/inferenceserver/$REPO_VERSION/tf_model_store"}
EXPORT_PATH="`pwd`/results"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
TRITON_LAUNCH_MODE=${TRITON_LAUNCH_MODE:="local"}
CLIENT_PROTOCOL="grpc"
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CHECKPOINT_DIRECTORY="`pwd`/checkpoints"

rm -rf $OUTPUT_MODEL_REPOSITORY

python3 test_config_generator.py --model-names resnet_v1_50_cpu_graphdef

# Compute expected columns (2 instance count * conccurrency * 7 dynamic batch size)
let "TEST_OUTPUT_NUM_ROWS = 42"

RET=0

# Run the profiler
set +e
PROFILE_CONFIG='config-profile.yml'
MODEL_ANALYZER_ARGS="-m $MODEL_REPOSITORY --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE --checkpoint-directory $CHECKPOINT_DIRECTORY -f $PROFILE_CONFIG"
MODEL_ANALYZER_SUBCOMMAND="profile"
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
fi

# Run analyze to generate reports
MODEL_ANALYZER_ARGS="--analysis-models resnet_v1_50_cpu_graphdef -e $EXPORT_PATH --filename-server-only=$FILENAME_SERVER_ONLY  --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS "
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
    INFERENCE_NUM_COLUMNS=9

    check_table_row_column \
        $ANALYZER_LOG "" "" \
        $MODEL_METRICS_INFERENCE_FILE "" "" \
        $INFERENCE_NUM_COLUMNS $TEST_OUTPUT_NUM_ROWS \
        0 0 0 0
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed.\n***"
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
