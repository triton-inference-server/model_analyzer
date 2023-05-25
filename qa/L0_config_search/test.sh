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

source ../common/util.sh
source ../common/check_analyzer_results.sh
create_logs_dir "L0_config_search"

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/nvdl/datasets/inferenceserver/$REPO_VERSION/libtorch_model_store"}
MODEL_NAMES="vgg19_libtorch"
EXPORT_PATH="`pwd`/results"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
TRITON_LAUNCH_MODES="local remote"
CLIENT_PROTOCOL="grpc"
PORTS=(`find_available_ports 3`)
http_port="${PORTS[0]}"
grpc_port="${PORTS[1]}"
metrics_port="${PORTS[2]}"
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CHECKPOINT_DIRECTORY="`pwd`/checkpoints"
MODEL_ANALYZER_PROFILE_BASE_ARGS="--model-repository $MODEL_REPOSITORY"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS  --filename-server-only=$FILENAME_SERVER_ONLY"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"
MODEL_ANALYZER_GLOBAL_OPTIONS="-v"

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
        rm -rf $OUTPUT_MODEL_REPOSITORY

        TEST_NAME=$(basename "$config" | sed 's/\.[^.]*$//')_${launch_mode}
        create_result_paths -test-name $TEST_NAME
        SERVER_LOG=${TEST_LOG_DIR}/server.${TEST_NAME}.log

        set +e

        MODEL_ANALYZER_PORTS="--triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]} --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $config --triton-launch-mode $launch_mode -e $EXPORT_PATH --checkpoint-directory $CHECKPOINT_DIRECTORY"
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS $MODEL_ANALYZER_PORTS"

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

        echo -e "\n*** Re-running profile\n***"
        ANALYZER_LOG=${TEST_LOG_DIR}/analyzer.${TEST_NAME}_rerun.log
        run_analyzer
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
            cat $ANALYZER_LOG
            RET=1
        fi

        if [ $launch_mode == 'remote' ]; then
            kill $SERVER_PID
            wait $SERVER_PID
        fi

        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS"
        
        SERVER_METRICS_FILE=${EXPORT_PATH}/results/${FILENAME_SERVER_ONLY}
        MODEL_METRICS_GPU_FILE=${EXPORT_PATH}/results/${FILENAME_GPU_MODEL}
        MODEL_METRICS_INFERENCE_FILE=${EXPORT_PATH}/results/${FILENAME_INFERENCE_MODEL}
        METRICS_NUM_COLUMNS=10
        INFERENCE_NUM_COLUMNS=9
        SERVER_METRICS_NUM_COLUMNS=5
        
        # Check that rerun skipped getting server metrics
        grep "GPU devices match checkpoint" $ANALYZER_LOG | wc -l
        if [ $? -eq 0]; then
            echo -e "\n***\n*** Test Verification Failed - GPU devices did not match checkpoint on rerun.\n***"
            cat $ANALYZER_LOG
            RET=1
        fi

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

        # Check that GPUs don't match and we return an error
        if [ $launch_mode != 'remote' ]; then
            sed -i 's/GPU-/GPU-1-/g' $CHECKPOINT_DIRECTORY/1.ckpt
            
            run_analyzer
            if [ $? -e 0 ]; then
                echo -e "\n***\n*** Test Output Verification Failed. model-analyzer exited sucessfully, but GPUs did not match checkpoint. \n***"
                cat $ANALYZER_LOG
                RET=1
            fi
            grep "GPU devices do not match checkpoint" $ANALYZER_LOG | wc -l
            if [ $? -eq 0]; then
                echo -e "\n***\n*** Test Verification Failed - GPU did not mismatch checkpoint on rerun.\n***"
                cat $ANALYZER_LOG
                RET=1
            fi
        fi
        set -e
    done
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
