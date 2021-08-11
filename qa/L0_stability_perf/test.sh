# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
PERF_LOG_BASE="perf.log"
source ../common/util.sh

rm -f *.log

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
PERF_ANALYZER="`which perf_analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/model_analyzer_benchmark_models"}
TRITON_LAUNCH_MODE=${TRITON_LAUNCH_MODE:="local"}
CLIENT_PROTOCOL="grpc"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CONCURRENCY="1"
BENCHMARK_MODELS="bert_savedmodel resnet50_fp32_libtorch"
MODEL_NAMES="$(echo $BENCHMARK_MODELS | sed 's/ /,/g')"
CHECKPOINT_DIRECTORY="./checkpoints"

METRIC_TOLERANCE_PERCENT=${METRIC_TOLERANCE_PERCENT:=5}
MEASUREMENT_REQUEST_COUNT=${MEASUREMENT_REQUEST_COUNT:=500}

# Generate test configs
python3 test_config_generator.py --profile-models $MODEL_NAMES --request-count $MEASUREMENT_REQUEST_COUNT

LIST_OF_CONFIG_FILES=(`ls | grep .yaml`)

if [ ${#LIST_OF_CONFIG_FILES[@]} -le 0 ]; then
    echo -e "\n***\n*** Test Failed. No config file exists. \n***"
    exit 1
fi

# Set analyzer config options
RET=0

MODEL_ANALYZER_BASE_ARGS="-m $MODEL_REPOSITORY --concurrency $CONCURRENCY"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_SUBCOMMAND="profile"

for CONFIG_FILE in ${LIST_OF_CONFIG_FILES[@]}; do
    LOG_PREFIX=${CONFIG_FILE#"config-"}
    MODEL_NAME=${LOG_PREFIX%".yaml"}
    ANALYZER_LOG=${MODEL_NAME}.${ANALYZER_LOG_BASE}
    PERF_LOG=${MODEL_NAME}.${PERF_LOG_BASE}
    
    set +e 

    # Run the model analyzer
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS -f $CONFIG_FILE"
    run_analyzer
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
        cat $ANALYZER_LOG
        RET=1
    fi

    set -e

    # Now run the tritonserver and perf_analyzer
    SERVER=/opt/tritonserver/bin/tritonserver
    SERVER_ARGS="--model-repository=${MODEL_REPOSITORY} --model-control-mode=explicit --load-model $MODEL_NAME"
    SERVER_ARGS="$SERVER_ARGS --http-port ${PORTS[0]} --grpc-port ${PORTS[1]} --metrics-port ${PORTS[2]}"
    SERVER_HTTP_PORT=${PORTS[0]}

   
    for concurrency in "$(echo $CONCURRENCY | sed 's/,/ /g')"; do
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        set +e

        PERF_ANALYZER_ARGS="-i $CLIENT_PROTOCOL -u localhost:${PORTS[1]} -m $MODEL_NAME --concurrency-range $concurrency --percentile 95"
        PERF_ANALYZER_ARGS="$PERF_ANALYZER_ARGS --measurement-mode count_windows --measurement-request-count $MEASUREMENT_REQUEST_COUNT"
        
        $PERF_ANALYZER $PERF_ANALYZER_ARGS >> $PERF_LOG 2>&1
        if [ ! $? -eq 0 ]; then
            cat $PERF_LOG
            echo -e "\n***\n*** Test Failed: perf_analyzer return non-zero exit code for model $MODEL_NAME. \n***"
            RET=1
        fi

        set -e
        # Kill triton
        kill $SERVER_PID
        wait $SERVER_PID 
    done
done

rm -f $SERVER_LOG

set +e
# Check the Analyzer log for correct output
TEST_NAME='perf_stability'
python3 check_results.py -t $TEST_NAME --tolerance $METRIC_TOLERANCE_PERCENT
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
    cat $ANALYZER_LOG
    RET=1
fi

set -e

rm -rf $CHECKPOINT_DIRECTORY/*
rm *.yaml

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
