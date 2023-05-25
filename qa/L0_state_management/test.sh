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

source ../common/util.sh
create_logs_dir "L0_state_management"

rm -rf *.yml

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/nvdl/datasets/inferenceserver/$REPO_VERSION/libtorch_model_store"}
QA_MODELS="vgg19_libtorch resnet50_libtorch libtorch_amp_resnet50"
MODEL_NAMES="$(echo $QA_MODELS | sed 's/ /,/g')"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
TRITON_LAUNCH_MODE="local"
CLIENT_PROTOCOL="http"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}

MODEL_ANALYZER_PROFILE_BASE_ARGS="-m $MODEL_REPOSITORY --run-config-search-disable"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_SUBCOMMAND="profile"
MODEL_ANALYZER_GLOBAL_OPTIONS="-v"

python3 test_config_generator.py -m $MODEL_NAMES

RET=0

# TEST CASE: Run the config and count the number of checkpoints
TEST_NAME="num_checkpoints"
CONFIG_FILE="config-single.yml"

# Create new EXPORT_PATH, CHECKPOINT_DIRECTORY and ANALYZER_LOG
create_result_paths -test-name $TEST_NAME

set +e

MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE --checkpoint-directory $CHECKPOINT_DIRECTORY -e $EXPORT_PATH"
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

# TEST CASE: Run the config again and make sure that no perf analyzer runs took place
TEST_NAME="loading_checkpoints"

# Create new EXPORT_PATH and ANALYZER_LOG
create_result_paths -test-name $TEST_NAME -checkpoints false
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE --checkpoint-directory $CHECKPOINT_DIRECTORY -e $EXPORT_PATH"

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

# TEST CASE: run config multple and send SIGINT after 2 models run
TEST_NAME="interrupt_handling"

# Create new EXPORT_PATH, CHECKPOINT_DIRECTORY and ANALYZER_LOG
create_result_paths -test-name $TEST_NAME

CONFIG_FILE="config-multi.yml"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE --checkpoint-directory $CHECKPOINT_DIRECTORY -e $EXPORT_PATH"

set +e
run_analyzer_nohup
ANALYZER_PID=$!

sleep 5
until [[ $(ls $CHECKPOINT_DIRECTORY | wc -l) == "1" ]]; do
    sleep 1
done

kill -2 $ANALYZER_PID
wait $ANALYZER_PID

if [ $? -ne 0 ]; then
    python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -d $CHECKPOINT_DIRECTORY -l $ANALYZER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi
else
    echo -e "\n***\n*** Test Failed. model-analyzer exited with ZERO exit code when SIGINT occurred. \n***"
    cat $ANALYZER_LOG
    RET=1
    
fi

# Create new EXPORT_PATH and ANALYZER_LOG
TEST_NAME="continue_after_checkpoint"
create_result_paths -test-name $TEST_NAME -checkpoints false
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE --checkpoint-directory $CHECKPOINT_DIRECTORY -e $EXPORT_PATH"

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

# TEST CASE: Run config-multiple and send 3 SIGINT after server is profiled
TEST_NAME="early_exit"

# Create new EXPORT_PATH, CHECKPOINT_DIRECTORY and ANALYZER_LOG
create_result_paths -test-name $TEST_NAME

CONFIG_FILE="config-multi.yml"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE --checkpoint-directory $CHECKPOINT_DIRECTORY -e $EXPORT_PATH"

set +e
run_analyzer_nohup
ANALYZER_PID=$!

sleep 2
until [[ ! -z `grep "Stopped Triton Server." $ANALYZER_LOG` ]]; do
    sleep 1
done

until [[ "`grep 'SIGINT' $ANALYZER_LOG | wc -l`" -gt "3" ]]; do
    kill -2 $ANALYZER_PID
    sleep 0.5
done
wait $ANALYZER_PID

if [ $? -ne 1 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer did not exit with expected exit code (1). \n***"
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

# TEST CASE: Have results mixed across runs
TEST_NAME="measurements_consistent_with_config_1"

# Create new EXPORT_PATH, CHECKPOINT_DIRECTORY and ANALYZER_LOG
create_result_paths -test-name $TEST_NAME

CONFIG_FILE="config-mixed-first.yml"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE --checkpoint-directory $CHECKPOINT_DIRECTORY -e $EXPORT_PATH"

set +e
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
fi

TEST_NAME="measurements_consistent_with_config"
CONFIG_FILE="config-mixed-second.yml"

# Create new EXPORT_PATH and ANALYZER_LOG
create_result_paths -test-name ${TEST_NAME}_2 -checkpoints false
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $CONFIG_FILE --checkpoint-directory $CHECKPOINT_DIRECTORY -e $EXPORT_PATH"

run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
fi

python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -d $CHECKPOINT_DIRECTORY -l $ANALYZER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
    cat $ANALYZER_LOG
    RET=1
fi

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
