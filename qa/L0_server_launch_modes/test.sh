# Copyright (c) 2020,21 NVIDIA CORPORATION. All rights reserved.
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

rm -f $LOGS_DIR/*.log
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
rm -rf $OUTPUT_MODEL_REPOSITORY

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
TRITON_DOCKER_IMAGE=${TRITONSERVER_BASE_IMAGE_NAME}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/nvdl/datasets/inferenceserver/$REPO_VERSION/libtorch_model_store"}
MODEL_NAMES="vgg19_libtorch"
TRITON_LAUNCH_MODES="local docker remote c_api"
CLIENT_PROTOCOLS="http grpc"
PORTS=(`find_available_ports 3`)
http_port="${PORTS[0]}"
grpc_port="${PORTS[1]}"
metrics_port="${PORTS[2]}"
GPUS=(`get_all_gpus_uuids`)
MODEL_ANALYZER_BASE_ARGS="-m $MODEL_REPOSITORY --profile-models $MODEL_NAMES"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-http-endpoint localhost:$http_port --triton-grpc-endpoint localhost:$grpc_port"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-metrics-url http://localhost:$metrics_port/metrics"

python3 test_config_generator.py --protocols "`echo $CLIENT_PROTOCOLS | sed 's/ /,/g'`" --launch-modes "`echo $TRITON_LAUNCH_MODES | sed 's/ /,/g'`"

LIST_OF_CONFIG_FILES=(`ls | grep .yaml`)

if [ ${#LIST_OF_CONFIG_FILES[@]} -le 0 ]; then
    echo -e "\n***\n*** Test Failed. No config file exists. \n***"
    exit 1
fi

# Run the model-analyzer, both client protocols
RET=0

function convert_gpu_array_to_flag() {
    gpu_array=($@)
    if [ "$gpu_array" == "empty_gpu_flag" ]; then
        echo "--gpus []"
    elif [ ! -z "${gpu_array[0]}" ]; then
        gpus_flag="--gpus "
        for gpu in ${gpu_array[@]}; do
            gpus_flag="${gpus_flag}${gpu},"
        done
        # Remove trailing comma
        gpus_flag=${gpus_flag::-1}
        echo $gpus_flag
    else
        echo ""
    fi
}

function run_server_launch_modes() {
    # Login to the GitLab CI. Required for pulling the Triton container.
    docker login -u gitlab-ci-token -p "${CI_JOB_TOKEN}" "${CI_REGISTRY}"

    gpus=($@)
    for CONFIG_FILE in ${LIST_OF_CONFIG_FILES[@]}; do
        # e.g. config-docker-http.yaml -> config-docker-http
        CONFIG_PARAMETERS=${CONFIG_FILE%".yaml"}

        # config-docker-http -> [config, docker, http]
        PARAMETERS=(${CONFIG_PARAMETERS//-/ })
        LAUNCH_MODE=${PARAMETERS[1]}
        PROTOCOL=${PARAMETERS[2]}
        
        ANALYZER_LOG=$LOGS_DIR/analyzer.${LAUNCH_MODE}.${PROTOCOL}.log
        SERVER_LOG=$LOGS_DIR/${LAUNCH_MODE}.${PROTOCOL}.server.log

        MODEL_ANALYZER_GLOBAL_OPTIONS="-v"
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS `convert_gpu_array_to_flag ${gpus[@]}` -f $CONFIG_FILE"
        MODEL_CONTROL_MODE=''
        RELOAD_MODEL_DISABLE=''
        MA_EXPECTED_RESULT='EP'     # Expected Pass

        _run_single_config
        if [ $? -ne 0 ]; then
            exit 1
        fi
    done
}

function _run_single_config() {
    # Set arguments for various launch modes
    if [ "$LAUNCH_MODE" == "remote" ]; then    

        # EP = "expected pass"
        # EF = "expected fail"
        model_mode_combos=(
            '--model-control-mode=explicit;;EP'
            ';--reload-model-disable;EP'
            '--model-control-mode=explicit;--reload-model-disable;EF' 
            # The following 'expected fail' test is commented-out since the intended 
            # server exception does not cause MA to return a non-zero exit status
            # ';;EF'   
            )

        for model_mode_combo in ${model_mode_combos[@]}
        do
            IFS=';' read -ra model_mode <<< "$model_mode_combo"
            length=${#model_mode[@]}
            # trailing spaces are omitted from array
            if [ $length -ne 3 ]; then
                echo -e "\n***\n*** Array setup incorrectly\n***"
                exit 1
            fi

            for i in ${!model_mode[@]}; do
                if [ $i -eq 0 ]; then
                    MODEL_CONTROL_MODE=${model_mode[$i]}
                elif [ $i -eq 1 ]; then
                    RELOAD_MODEL_DISABLE=${model_mode[$i]}
                elif [ $i -eq 2 ]; then
                    MA_EXPECTED_RESULT=${model_mode[$i]}
                fi
            done

            # For remote launch, set server args and start server
            SERVER=`which tritonserver`
            SERVER_ARGS="--model-repository=$MODEL_REPOSITORY $MODEL_CONTROL_MODE --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port"
            SERVER_HTTP_PORT=${http_port}
    
            run_server
            if [ "$SERVER_PID" == "0" ]; then
                echo -e "\n***\n*** Failed to start $SERVER\n***"
                cat $SERVER_LOG
                exit 1
            fi

            MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS $RELOAD_MODEL_DISABLE `convert_gpu_array_to_flag ${gpus[@]}` -f $CONFIG_FILE"
            _run_analyzer_and_check_results
            if [ $? -ne 0 ]; then
                return 1
            fi        
        done

        return

    elif [ "$LAUNCH_MODE" == "c_api" ]; then
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --perf-output-path=${SERVER_LOG}"
    elif [ "$LAUNCH_MODE" == "docker" ]; then
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-output-path=${SERVER_LOG} --triton-docker-image=$TRITON_SERVER_CONTAINER_IMAGE_NAME"
    else
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-output-path=${SERVER_LOG}"
    fi

    _run_analyzer_and_check_results
    if [ $? -ne 0 ]; then
        return 1
    fi        
}

function _run_analyzer_and_check_results() {
    # Run the analyzer and check the results, enough to just profile the server
    set +e
    MODEL_ANALYZER_SUBCOMMAND="profile"
    run_analyzer
    MA_ACTUAL_RESULT=$?
    _check_analyzer_exit_status

    if [ "$LAUNCH_MODE" == "remote" ]; then
        kill $SERVER_PID
        wait $SERVER_PID
    else
        if [ ! -s "$SERVER_LOG" ]; then
            echo -e "\n***\n*** Test Output Verification Failed : No logs found\n***"
            cat $ANALYZER_LOG
            RET=1
            exit 1
        fi
    fi

    if [ "$gpus" == "empty_gpu_flag" ]; then
        python3 check_gpus.py --analyzer-log $ANALYZER_LOG
    elif [ -z "$gpus" ]; then
        python3 check_gpus.py --analyzer-log $ANALYZER_LOG --gpus `echo ${GPUS[@]} | sed "s/ /,/g"` --check-visible
    else
        python3 check_gpus.py --analyzer-log $ANALYZER_LOG --gpus `echo ${gpus[@]} | sed "s/ /,/g"`
    fi
    if [ $? -ne 0 ]; then
        RET=1
        return 1
    fi
    set -e

    rm -rf $OUTPUT_MODEL_REPOSITORY
    rm -rf checkpoints && mkdir checkpoints
}

function _check_analyzer_exit_status() {
    if [ "$MA_EXPECTED_RESULT" == "EP" ]; then
        # Expected Pass
        if [ "$MA_ACTUAL_RESULT" != "0" ]; then
            echo -e "\n***\n*** Test with launch mode '${LAUNCH_MODE}' using ${PROTOCOL} client Failed."\
                    "\n***     model-analyzer exited with non-zero exit code (${MA_ACTUAL_RESULT}). \n***"
            cat $ANALYZER_LOG
            RET=1
            exit 1
        fi
    elif [ "$MA_EXPECTED_RESULT" == "EF" ]; then
        # Expected fail
        if [ "$MA_ACTUAL_RESULT" == "0" ]; then
            echo -e "\n***\n*** Test with launch mode '${LAUNCH_MODE}' using ${PROTOCOL} should have Failed."\
                    "\n***     model-analyzer exited with zero exit code. (${MA_ACTUAL_RESULT})\n***"
            cat $ANALYZER_LOG
            RET=1
            exit 1
        fi
    else
        echo -e "\n***\n*** MA_EXPECTED_RESULT not setup properly. MA_EXPECTED_RESULT=${MA_EXPECTED_RESULT}\n***"
        RET=1
        exit 1
    fi
}

# This test will be executed with 4-GPUs.
CUDA_DEVICE_ORDER="PCI_BUS_ID"

##########################################################
# Test controling the GPUs with the CUDA_VISIBLE_DEVICES #
##########################################################
export CUDA_VISIBLE_DEVICES=3
run_server_launch_modes

export CUDA_VISIBLE_DEVICES=1,2
run_server_launch_modes

export CUDA_VISIBLE_DEVICES=0,1,2
run_server_launch_modes

unset CUDA_VISIBLE_DEVICES

#################################################
# Test controling the GPUs with the --gpus flag #
#################################################

CURRENT_GPUS=(${GPUS[2]})
run_server_launch_modes "$CURRENT_GPUS"

CURRENT_GPUS=${GPUS[@]:1}
run_server_launch_modes "$CURRENT_GPUS"

CURRENT_GPUS="empty_gpu_flag"
run_server_launch_modes "$CURRENT_GPUS"

# Test with GPU ID
CURRENT_GPUS="0"
run_server_launch_modes "$CURRENT_GPUS"

CURRENT_GPUS="1 2"
run_server_launch_modes "$CURRENT_GPUS"

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
