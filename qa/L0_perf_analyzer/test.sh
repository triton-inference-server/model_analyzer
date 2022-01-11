# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ANALYZER_LOG="test.log"
source ../common/util.sh

rm -f *.log

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/nvdl/datasets/inferenceserver/$REPO_VERSION/libtorch_model_store"}
BATCH_SIZES="1"
CONCURRENCY="1"
TRITON_LAUNCH_MODE="local"
CLIENT_PROTOCOL="http"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CHECKPOINT_DIRECTORY="`pwd`/checkpoints"

rm -rf *.yml
rm -rf $CHECKPOINT_DIRECTORY && mkdir -p $CHECKPOINT_DIRECTORY

MODEL_ANALYZER_PROFILE_BASE_ARGS="-m $MODEL_REPOSITORY -b $BATCH_SIZES -c $CONCURRENCY --run-config-search-disable"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics --perf-analyzer-cpu-util=100000"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_SUBCOMMAND="profile"

python3 test_config_generator.py

LIST_OF_CONFIG_FILES=(`ls | grep .yml`)

if [ ${#LIST_OF_CONFIG_FILES[@]} -le 0 ]; then
    echo -e "\n***\n*** Test Failed. No config file exists. \n***"
    RET=1
    exit $RET
fi

# Run the analyzer with perf-measurement-window=5000ms and expect no adjustment
RET=0

for CONFIG_FILE in ${LIST_OF_CONFIG_FILES[@]}; do
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE"
    if [[ "$CONFIG_FILE" == "config-time-no-adjust.yml" ]]; then
        TEST_NAME="time_window"
    elif [[ "$CONFIG_FILE" == "config-time-adjust.yml" ]]; then
        TEST_NAME="time_window_adjust"
    elif [[ "$CONFIG_FILE" == "config-count-no-adjust.yml" ]]; then
        TEST_NAME="count_window"
    elif [[ "$CONFIG_FILE" == "config-additive-args-count-no-adjust.yml" ]]; then
        TEST_NAME="count_window"
    elif [[ "$CONFIG_FILE" == "config_perf_output_timeout.yml" ]]; then
        TEST_NAME="perf_output_timeout"
    fi

    echo $TEST_NAME

    set +e

    run_analyzer
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
        cat $ANALYZER_LOG
        RET=1
    else
        # Check for the correct output
        python3 check_results.py -f $CONFIG_FILE --test-name $TEST_NAME --analyzer-log $ANALYZER_LOG
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed.\n***"
            cat $ANALYZER_LOG
            RET=1
        fi
    fi

    rm -f $ANALYZER_LOG
    rm -f checkpoints/*

    set -e
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
