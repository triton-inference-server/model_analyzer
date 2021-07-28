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
source ../common/check_analyzer_results.sh

rm -f *.log
rm -rf results && mkdir -p results

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/$REPO_VERSION/libtorch_model_store"}
MODEL_NAMES="vgg19_libtorch"
EXPORT_PATH="`pwd`/results"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
TRITON_LAUNCH_MODES="remote local"
CLIENT_PROTOCOL="grpc"
PORTS=(`find_available_ports 3`)
http_port="${PORTS[0]}"
grpc_port="${PORTS[1]}"
metrics_port="${PORTS[2]}"
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CHECKPOINT_DIRECTORY="`pwd`/checkpoints"
MODEL_ANALYZER_PROFILE_BASE_ARGS="--model-repository $MODEL_REPOSITORY --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"

MODEL_ANALYZER_ANALYZE_BASE_ARGS="--analysis-models $MODEL_NAMES -e $EXPORT_PATH --filename-server-only=$FILENAME_SERVER_ONLY --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_ANALYZE_BASE_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"

python3 test_config_generator.py -m $MODEL_NAMES

LIST_OF_CONFIG_FILES=(`ls | grep .yml`)

RET=0

if [ ${#LIST_OF_CONFIG_FILES[@]} -le 0 ]; then
    echo -e "\n***\n*** Test Failed. No config file exists. \n***"
    exit 1
fi

for launch_mode in $TRITON_LAUNCH_MODES; do

    # Run the analyzer with various configurations and check the results
    for config in ${LIST_OF_CONFIG_FILES[@]}; do
        rm -rf results && mkdir -p results && rm -rf $OUTPUT_MODEL_REPOSITORY && rm -rf $CHECKPOINT_DIRECTORY/*
        set +e

        MODEL_ANALYZER_PORTS="--triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]} --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $config --triton-launch-mode $launch_mode"
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS $MODEL_ANALYZER_PORTS"

        ANALYZER_LOG=analyzer.${launch_mode}.${config}.log

        if [ $launch_mode == 'remote' ]; then
            NUM_ROW_OUTPUT_FILE=`echo $config | sed 's/\.yml//'`-param-$launch_mode.txt
            NUM_MODELS_OUTPUT_FILE=`echo $config | sed 's/\.yml//'`-models-$launch_mode.txt

            # For remote launch, set server args and start server
            SERVER=`which tritonserver`
            SERVER_ARGS="--model-repository=$MODEL_REPOSITORY --model-control-mode=explicit --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port"
            SERVER_HTTP_PORT="${http_port}"

            run_server
            if [ "$SERVER_PID" == "0" ]; then
                echo -e "\n***\n*** Failed to start $SERVER\n***"
                cat $SERVER_LOG
                RET=1
            fi
        else
            NUM_MODELS_OUTPUT_FILE=`echo $config | sed 's/\.yml//'`-models.txt
            NUM_ROW_OUTPUT_FILE=`echo $config | sed 's/\.yml//'`-param.txt
        fi

        TEST_OUTPUT_NUM_ROWS=`cat $NUM_ROW_OUTPUT_FILE`
        TEST_MODELS_NUM=`cat $NUM_MODELS_OUTPUT_FILE`
        MODEL_ANALYZER_SUBCOMMAND="profile"
        run_analyzer
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
            cat $ANALYZER_LOG
            RET=1
        fi
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS"
        MODEL_ANALYZER_SUBCOMMAND="analyze"
        run_analyzer
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
            cat $ANALYZER_LOG
            RET=1
        else
            if [ $launch_mode == 'remote' ]; then
                kill $SERVER_PID
                wait $SERVER_PID
            fi
            SERVER_METRICS_FILE=${EXPORT_PATH}/results/${FILENAME_SERVER_ONLY}
            MODEL_METRICS_GPU_FILE=${EXPORT_PATH}/results/${FILENAME_GPU_MODEL}
            MODEL_METRICS_INFERENCE_FILE=${EXPORT_PATH}/results/${FILENAME_INFERENCE_MODEL}
            METRICS_NUM_COLUMNS=11
            INFERENCE_NUM_COLUMNS=10
            SERVER_METRICS_NUM_COLUMNS=5
            
            check_table_row_column \
                $ANALYZER_LOG $ANALYZER_LOG $ANALYZER_LOG \
                $MODEL_METRICS_INFERENCE_FILE $MODEL_METRICS_GPU_FILE $SERVER_METRICS_FILE \
                $INFERENCE_NUM_COLUMNS $TEST_OUTPUT_NUM_ROWS \
                $METRICS_NUM_COLUMNS $(($TEST_OUTPUT_NUM_ROWS * ${#GPUS[@]})) \
                $SERVER_METRICS_NUM_COLUMNS $((1 * ${#GPUS[@]}))
            if [ $? -ne 0 ]; then
                echo -e "\n***\n*** Test Output Verification Failed.\n***"
                cat $ANALYZER_LOG
                RET=1
            fi
        fi

        rm $ANALYZER_LOG
        rm -rf $EXPORT_PATH/*
        set -e
    done
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
