# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

ANALYZER_LOG_BASE="test.log"
source ../common/util.sh

rm -f *.log && rm -rf *.yml

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/$REPO_VERSION/libtorch_model_store"}
QA_MODELS="vgg19_libtorch resnet50_libtorch libtorch_amp_resnet50"
MODEL_NAMES="$(echo $QA_MODELS | sed 's/ /,/g')"
EXPORT_PATH="`pwd`/results"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
TRITON_LAUNCH_MODE="local"
CLIENT_PROTOCOL="http"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
rm -rf $OUTPUT_MODEL_REPOSITORY
CHECKPOINT_DIRECTORY=$EXPORT_PATH/checkpoints

MODEL_ANALYZER_PROFILE_BASE_ARGS="-m $MODEL_REPOSITORY --run-config-search-disable --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_SUBCOMMAND="profile"

python3 test_config_generator.py -m $MODEL_NAMES

RET=0

rm -rf $EXPORT_PATH && mkdir -p $EXPORT_PATH

# First run the config and count the number of checkpoints
TEST_NAME="num_checkpoints"
ANALYZER_LOG="${TEST_NAME}.${ANALYZER_LOG_BASE}"
CONFIG_FILE="config-single.yml"

set +e

MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE"
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -d $CHECKPOINT_DIRECTORY -l $ANALYZER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi
fi
set -e

# Second run the config again and make sure that no perf analyzer runs took place
TEST_NAME="loading_checkpoints"
ANALYZER_LOG="${TEST_NAME}.${ANALYZER_LOG_BASE}"

set +e
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -d $CHECKPOINT_DIRECTORY -l $ANALYZER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi
fi
set -e

# Clear checkpoints and results
rm -rf $EXPORT_PATH/*

# Fourth run config multple and send SIGINT after 2 models run
TEST_NAME="interrupt_handling"
CONFIG_FILE="config-multi.yml"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE"
ANALYZER_LOG="${TEST_NAME}.${ANALYZER_LOG_BASE}"

set +e
run_analyzer_nohup
ANALYZER_PID=$!

sleep 5
until [[ $(ls $EXPORT_PATH/checkpoints | wc -l) == "1" ]]; do
    sleep 1
done

kill -2 $ANALYZER_PID
wait $ANALYZER_PID

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -d $CHECKPOINT_DIRECTORY -l $ANALYZER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi
fi

TEST_NAME="continue_after_checkpoint"
ANALYZER_LOG="${TEST_NAME}.${ANALYZER_LOG_BASE}"

run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -d $CHECKPOINT_DIRECTORY -l $ANALYZER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi
fi
set -e

# Clear checkpoints and results
rm -rf $EXPORT_PATH/*

# For the fifth test we will have results mixed across runs
CONFIG_FILE="config-mixed-first.yml"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE"
ANALYZER_LOG="measurement_consistent_prep.${ANALYZER_LOG_BASE}"

set +e
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
fi

TEST_NAME="measurements_consistent_with_config"
CONFIG_FILE="config-mixed-second.yml"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE"
ANALYZER_LOG="${TEST_NAME}.${ANALYZER_LOG_BASE}"

run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
fi

# Run analyze because here we check the results not just the logs
MODEL_ANALYZER_SUBCOMMAND="analyze"
MODEL_ANALYZER_ARGS="-e $EXPORT_PATH --checkpoint-directory $CHECKPOINT_DIRECTORY --summarize False"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --analysis-models $MODEL_NAMES"
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -d $CHECKPOINT_DIRECTORY -l $ANALYZER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi
fi

rm -rf $EXPORT_PATH/*

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
