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
rm -rf results && mkdir -p results

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/model_analyzer_clara_pipelines"}
QA_MODELS="`ls $MODEL_REPOSITORY | head -2`"
MODEL_NAMES="$(echo $QA_MODELS | sed 's/ /,/g')"
BATCH_SIZES="1,2"
CONCURRENCY="1,2"
EXPORT_PATH="`pwd`/results"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_MODEL="model-metrics.csv"
TRITON_LAUNCH_MODE="local"
CLIENT_PROTOCOL="grpc"
TEST_OUTPUT_NUM_COLUMNS=7
PORTS=(`find_available_ports 3`)

MODEL_ANALYZER_ARGS="-m $MODEL_REPOSITORY -n $MODEL_NAMES -b $BATCH_SIZES -c $CONCURRENCY"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --export -e $EXPORT_PATH --filename-server-only=$FILENAME_SERVER_ONLY --filename-model=$FILENAME_MODEL"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"

# Compute expected columns
NUM_MODELS=`echo $QA_MODELS | awk '{print NF}'`
NUM_BATCHES=`echo $BATCH_SIZES | awk -F ',' '{print NF}'`
NUM_CONCURRENCIES=`echo $CONCURRENCY | awk -F ',' '{print NF}'`
let "TEST_OUTPUT_NUM_ROWS = $NUM_MODELS * $NUM_BATCHES * $NUM_CONCURRENCIES"

# Run the analyzer and check the results
RET=0

set +e
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    # Verify results
    check_analyzer_output $ANALYZER_LOG $TEST_OUTPUT_NUM_ROWS $TEST_OUTPUT_NUM_COLUMNS
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed\n***"
        cat $ANALYZER_LOG
        RET=1
    fi
    SERVER_METRICS_FILE=${EXPORT_PATH}/${FILENAME_SERVER_ONLY}
    MODEL_METRICS_FILE=${EXPORT_PATH}/${FILENAME_MODEL}
    check_exported_metrics $SERVER_METRICS_FILE $MODEL_METRICS_FILE $TEST_OUTPUT_NUM_ROWS $TEST_OUTPUT_NUM_COLUMNS
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed\n***"
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
