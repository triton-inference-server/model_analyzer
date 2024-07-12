#!/bin/bash
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
create_logs_dir "L0_optuna_search"

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/opt/triton-model-analyzer/examples/quick-start"}
QA_MODELS="add_sub"
MODEL_NAMES="$(echo $QA_MODELS | sed 's/ /,/g')"
TRITON_LAUNCH_MODE=${TRITON_LAUNCH_MODE:="local"}
CLIENT_PROTOCOL="grpc"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CONFIG_FILE="config.yml"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"

rm -rf $OUTPUT_MODEL_REPOSITORY
create_result_paths

python3 test_config_generator.py --profile-models $MODEL_NAMES

# Run the analyzer and check the results
RET=0

set +e

MODEL_ANALYZER_ARGS="-m $MODEL_REPOSITORY -f $CONFIG_FILE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS -e $EXPORT_PATH --checkpoint-directory $CHECKPOINT_DIRECTORY --filename-server-only=$FILENAME_SERVER_ONLY"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --run-config-search-mode optuna"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --latency-budget 10"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --skip-summary-reports"
MODEL_ANALYZER_SUBCOMMAND="profile"
MODEL_ANALYZER_GLOBAL_OPTIONS="-v"

run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    # Check the Analyzer log for correct output
    TEST_NAME='profile_logs'
    python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -l $ANALYZER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi

    SERVER_METRICS_FILE=${EXPORT_PATH}/results/${FILENAME_SERVER_ONLY}
    MODEL_METRICS_GPU_FILE=${EXPORT_PATH}/results/${FILENAME_GPU_MODEL}
    MODEL_METRICS_INFERENCE_FILE=${EXPORT_PATH}/results/${FILENAME_INFERENCE_MODEL}

    for file in SERVER_METRICS_FILE, MODEL_METRICS_GPU_FILE, MODEL_METRICS_INFERENCE_FILE; do
        check_no_csv_exists $file
        if [ $? -ne 0 ]; then
          echo -e "\n***\n*** Test Output Verification Failed.\n***"
          cat $ANALYZER_LOG
          RET=1
        fi
    done
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
