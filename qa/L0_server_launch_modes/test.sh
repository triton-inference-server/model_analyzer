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

source ../common/util.sh

rm -f *.log

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/model_analyzer_clara_pipelines"}
MODEL_NAMES="classification_chestxray_v1"
BATCH_SIZES="4"
CONCURRENCY="4"
PORTS=(`find_available_ports 3`)
http_port="${PORTS[0]}"
grpc_port="${PORTS[1]}"
metrics_port="${PORTS[2]}"
MODEL_ANALYZER_BASE_ARGS="-m $MODEL_REPOSITORY -n $MODEL_NAMES -b $BATCH_SIZES -c $CONCURRENCY"
MODEL_ANALYZER_PORTS="--triton-http-endpoint localhost:$http_port --triton-grpc-endpoint localhost:$grpc_port"
MODEL_ANALYZER_PORTS="$MODEL_ANALYZER_PROTS --triton-metrics-url http://localhost:$metrics_port/metrics"
TRITON_LAUNCH_MODES="docker remote local"
TRITON_SERVER_VERSION="20.11-py3"
CLIENT_PROTOCOLS="http grpc"
TEST_OUTPUT_NUM_ROWS=1
TEST_OUTPUT_NUM_COLUMNS=7

# Run the model-analyzer, both client protocols
RET=0

for PROTOCOL in $CLIENT_PROTOCOLS; do
    MODEL_ANALYZER_ARGS_WITH_PROTOCOL="$MODEL_ANALYZER_BASE_ARGS --client-protocol=$PROTOCOL"
    for LAUNCH_MODE in $TRITON_LAUNCH_MODES; do
        MODEL_ANALYZER_ARGS_WITH_LAUNCH_MODE="$MODEL_ANALYZER_ARGS_WITH_PROTOCOL --triton-launch-mode=$LAUNCH_MODE"
        ANALYZER_LOG=analyzer.${LAUNCH_MODE}.${PROTOCOL}.log

        # Set arguments for various launch modes
        if [ "$LAUNCH_MODE" == "local" ]; then    
            MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS_WITH_LAUNCH_MODE $MODEL_ANALYZER_PORTS"
        elif [ "$LAUNCH_MODE" == "docker" ]; then
            MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS_WITH_LAUNCH_MODE --triton-version=$TRITON_SERVER_VERSION $MODEL_ANALYZER_PORTS"
        elif [ "$LAUNCH_MODE" == "remote" ]; then
            MODEL_ANALYZER_PORTS="--triton-http-endpoint localhost:8000 --triton-grpc-endpoint localhost:8001"
            MODEL_ANALYZER_PORTS="$MODEL_ANALYZER_PROTS --triton-metrics-url http://localhost:8002/metrics"
            MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS_WITH_LAUNCH_MODE $MODEL_ANALYZER_PORTS"

            # For remote launch, set server args and start server
            SERVER=`which tritonserver`
            SERVER_ARGS="--model-repository=$MODEL_REPOSITORY --model-control-mode=explicit"
            SERVER_LOG=${PROTOCOL}.server.log
        
            run_server
            if [ "$SERVER_PID" == "0" ]; then
                echo -e "\n***\n*** Failed to start $SERVER\n***"
                cat $SERVER_LOG
                exit 1
            fi
        fi

        # Run the analyzer and check the results
        set +e
        run_analyzer
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test with launch mode '${LAUNCH_MODE}' using ${PROTOCOL} client Failed."\
                    "\n***     model-analyzer exited with non-zero exit code. \n***"
            cat $ANALYZER_LOG
            RET=1
        else
            check_analyzer_output $ANALYZER_LOG $TEST_OUTPUT_NUM_ROWS $TEST_OUTPUT_NUM_COLUMNS
            if [ $? -ne 0 ]; then
                echo -e "\n***\n*** Test Output Verification Failed\n***"
                cat $ANALYZER_LOG
                RET=1
            fi
        fi
        set -e

        if [ "$LAUNCH_MODE" == "remote" ]; then
            kill $SERVER_PID
            wait $SERVER_PID
        fi
    done
done


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
