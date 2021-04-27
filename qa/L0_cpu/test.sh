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

ANALYZER_LOG="test.log"
source ../common/util.sh

rm -f *.log logs/
rm -rf results && mkdir -p results

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/data/inferenceserver/$REPO_VERSION/tf_model_store"}
EXPORT_PATH="`pwd`/logs"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
TRITON_LAUNCH_MODE=${TRITON_LAUNCH_MODE:="local"}
CLIENT_PROTOCOL="grpc"
PORTS=(`find_available_ports 3`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
mkdir $EXPORT_PATH
MODEL_ANALYZER_ARGS="-m $MODEL_REPOSITORY -n resnet_v1_50_cpu_graphdef --run-config-search-max-concurrency 4 --run-config-search-max-instance-count 2 --export-path logs"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS -e $EXPORT_PATH --filename-server-only=$FILENAME_SERVER_ONLY"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"

rm -rf $OUTPUT_MODEL_REPOSITORY

# Compute expected columns (2 instance count * conccurrency * 7 dynamic batch size)
let "TEST_OUTPUT_NUM_ROWS = 42"

# Run the analyzer and check the results
RET=0

set +e
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    SERVER_METRICS_FILE=${EXPORT_PATH}/results/${FILENAME_SERVER_ONLY}
    MODEL_METRICS_GPU_FILE=${EXPORT_PATH}/results/${FILENAME_GPU_MODEL}
    MODEL_METRICS_INFERENCE_FILE=${EXPORT_PATH}/results/${FILENAME_INFERENCE_MODEL}
    INFERENCE_NUM_COLUMNS=10

    check_log_table_row_column $ANALYZER_LOG $INFERENCE_NUM_COLUMNS $TEST_OUTPUT_NUM_ROWS "Models\ \(Inference\):"
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $ANALYZER_LOG.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi

    OUTPUT_TAG="Model"
    check_csv_table_row_column $MODEL_METRICS_INFERENCE_FILE $INFERENCE_NUM_COLUMNS $TEST_OUTPUT_NUM_ROWS $OUTPUT_TAG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $MODEL_METRICS_INFERENCE_FILE.\n***"
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
