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
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/model_analyzer_clara_pipelines"}
MODEL_NAMES="classification_chestxray_v1"
BATCH_SIZES="1"
CONCURRENCY="1"
TRITON_LAUNCH_MODE="local"
CLIENT_PROTOCOL="http"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}

MODEL_ANALYZER_BASE_ARGS="-m $MODEL_REPOSITORY -n $MODEL_NAMES -b $BATCH_SIZES -c $CONCURRENCY"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY"

# Run the analyzer with perf-measurement-window=1000ms and expect no adjustment
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS --perf-measurement-window=5000"

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

rm -f $ANALYZER_LOG

# Run the analyzer with perf-measurement-window=50ms and expect adjustment
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS --perf-measurement-window=50"

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

rm -f $ANALYZER_LOG

# Run the analyzer with no-perf-output and fail if output detected
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS --no-perf-output"

run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
elif [ ! -z `grep "*** Measurement Settings ***" ${ANALYZER_LOG}` ]; then
    echo -e "\n***\n*** Expected output to be silenced. \n***"
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
