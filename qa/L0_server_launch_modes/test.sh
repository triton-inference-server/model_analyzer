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

rm -f *.log
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
rm -rf $OUTPUT_MODEL_REPOSITORY

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/$REPO_VERSION/libtorch_model_store"}
CHECKPOINT_REPOSITORY=${CHECKPOINT_REPOSITORY:="/mnt/dldata/inferenceserver/model_analyzer_checkpoints"}
MODEL_NAMES="vgg19_libtorch"
TRITON_LAUNCH_MODES="local docker remote c_api"
CLIENT_PROTOCOLS="http grpc"
PORTS=(`find_available_ports 3`)
http_port="${PORTS[0]}"
grpc_port="${PORTS[1]}"
metrics_port="${PORTS[2]}"
GPUS=(`get_all_gpus_uuids`)
CHECKPOINT_DIRECTORY="`pwd`/checkpoints"
MODEL_ANALYZER_BASE_ARGS="-m $MODEL_REPOSITORY --profile-models $MODEL_NAMES"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-http-endpoint localhost:$http_port --triton-grpc-endpoint localhost:$grpc_port"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-metrics-url http://localhost:$metrics_port/metrics"

# mkdir $CHECKPOINT_DIRECTORY
# cp $CHECKPOINT_REPOSITORY/server_launch_modes.ckpt $CHECKPOINT_DIRECTORY/0.ckpt

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
    gpus=($@)
    for CONFIG_FILE in ${LIST_OF_CONFIG_FILES[@]}; do
        # e.g. config-docker-http.yaml -> config-docker-http
        CONFIG_PARAMETERS=${CONFIG_FILE%".yaml"}

        # config-docker-http -> [config, docker, http]
        PARAMETERS=(${CONFIG_PARAMETERS//-/ })
        LAUNCH_MODE=${PARAMETERS[1]}
        PROTOCOL=${PARAMETERS[2]}
        
        ANALYZER_LOG=analyzer.${LAUNCH_MODE}.${PROTOCOL}.log
        SERVER_LOG=${LAUNCH_MODE}.${PROTOCOL}.server.log

        MODEL_ANALYZER_GLOBAL_OPTIONS="-v"
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS `convert_gpu_array_to_flag ${gpus[@]}` -f $CONFIG_FILE"

        # Set arguments for various launch modes
        if [ "$LAUNCH_MODE" == "remote" ]; then    
            # For remote launch, set server args and start server
            SERVER=`which tritonserver`
            SERVER_ARGS="--model-repository=$MODEL_REPOSITORY --model-control-mode=explicit --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port"
            SERVER_HTTP_PORT=${http_port}
            
            run_server
            if [ "$SERVER_PID" == "0" ]; then
                echo -e "\n***\n*** Failed to start $SERVER\n***"
                cat $SERVER_LOG
                exit 1
            fi
        elif [ "$LAUNCH_MODE" == "c_api" ]; then
            # c_api does not get server only metrics, so for GPUs to appear in log, we must profile (delete checkpoint)
            rm -f $CHECKPOINT_DIRECTORY/*
            MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --perf-output-path=${SERVER_LOG}"
        else
            MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-output-path=${SERVER_LOG}"
        fi

        # Run the analyzer and check the results, enough to just profile the server
        set +e
        MODEL_ANALYZER_SUBCOMMAND="profile"
        run_analyzer
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test with launch mode '${LAUNCH_MODE}' using ${PROTOCOL} client Failed."\
                    "\n***     model-analyzer exited with non-zero exit code. \n***"
            cat $ANALYZER_LOG
            RET=1
        fi

        if [ "$LAUNCH_MODE" == "remote" ]; then
            kill $SERVER_PID
            wait $SERVER_PID
        else
            if [ ! -s "$SERVER_LOG" ]; then
                echo -e "\n***\n*** Test Output Verification Failed : No logs found\n***"
                cat $ANALYZER_LOG
                RET=1
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
            break
        fi
        set -e

        rm -rf $OUTPUT_MODEL_REPOSITORY
    done
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

rm -rf $CHECKPOINT_DIRECTORY

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
