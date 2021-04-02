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
CONFIG_FILE="config.yaml"
TRITON_LOG="triton_output.log"
BATCH_SIZES="1"
CONCURRENCY="1"
EXPORT_PATH="`pwd`/results"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
TRITON_LAUNCH_MODE="local"
TRITON_SERVER_VERSION="21.03-py3"
CLIENT_PROTOCOL="grpc"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}

MODEL_ANALYZER_ARGS="-m $MODEL_REPOSITORY -b $BATCH_SIZES -c $CONCURRENCY"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE --triton-version=$TRITON_SERVER_VERSION"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics --triton-output-path=${TRITON_LOG}"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY -f config.yaml --override-output-model-repository"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --run-config-search-disable"

rm -rf $OUTPUT_MODEL_REPOSITORY

# Run the analyzer and check the results
RET=0

set +e

run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    # Check the triton log for all flags
    STRICT_MODEL_CONFIG=(`awk '/strict_model_config/ {print $4}' ${TRITON_LOG}`)
    if [[ "$STRICT_MODEL_CONFIG" != "0" ]]; then
        echo -e "\n***\n*** Test Output Verification failed. strict-model-config flag not passed to Triton. \n***"
        echo $TRITON_LOG
        RET=1
    fi

    # Check analyzer log for perf flag
    P95_STABILIZATION=(`grep 'Stabilizing using p95 latency' ${ANALYZER_LOG}`)
    if [[ -z $P95_STABILIZATION ]]; then
        echo -e "\n***\n*** Test Output Verification failed. percentile flag not passed to perf_analyzer. \n***"
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
