# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/$REPO_VERSION/libtorch_model_store"}
BATCH_SIZES="1"
CONCURRENCY="1"
TRITON_LAUNCH_MODE="local"
CLIENT_PROTOCOL="http"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CHECKPOINT_DIRECTORY="`pwd`/checkpoints"
rm -rf *.yml
python3 config_generator.py

rm -rf $CHECKPOINT_DIRECTORY && mkdir -p $CHECKPOINT_DIRECTORY

MODEL_ANALYZER_PROFILE_BASE_ARGS="-m $MODEL_REPOSITORY -b $BATCH_SIZES -c $CONCURRENCY --run-config-search-disable"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics --perf-analyzer-cpu-util=100000"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_SUBCOMMAND="profile"

# Run the analyzer with perf-measurement-window=5000ms and expect no adjustment
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f config-time-window-5000.yml"

RET=0
set +e
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
elif [[ ! -z `grep "perf_analyzer's measurement window is too small" ${ANALYZER_LOG}` ]]; then
    echo -e "\n***\n*** Unexpected perf window adjustment was made. \n***"
    cat $ANALYZER_LOG
    RET=1
fi

rm -f $ANALYZER_LOG && rm -f checkpoints/*

# Run the analyzer with perf-measurement-window=50ms and expect adjustment
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f config-time-window-50.yml --perf-output=True"

run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
elif [[ -z `grep "measurement window is too small, increased to" ${ANALYZER_LOG}` ]]; then
    echo -e "\n***\n*** Expected perf window adjustment. \n***"
    cat $ANALYZER_LOG
    RET=1
fi

set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
