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
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
rm -rf $OUTPUT_MODEL_REPOSITORY

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/$REPO_VERSION/libtorch_model_store"}
MODEL_NAMES="vgg19_libtorch"
BATCH_SIZES="4"
CONCURRENCY="4"
PORTS=(`find_available_ports 3`)
http_port="${PORTS[0]}"
grpc_port="${PORTS[1]}"
metrics_port="${PORTS[2]}"
CHECKPOINT_DIRECTORY="`pwd`/checkpoints"
MODEL_ANALYZER_BASE_ARGS="-m $MODEL_REPOSITORY --profile-models $MODEL_NAMES -b $BATCH_SIZES -c $CONCURRENCY --run-config-search-disable --perf-analyzer-cpu-util 600"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_PORTS="--triton-http-endpoint localhost:$http_port --triton-grpc-endpoint localhost:$grpc_port"
MODEL_ANALYZER_PORTS="$MODEL_ANALYZER_PORTS --triton-metrics-url http://localhost:$metrics_port/metrics"
TRITON_LAUNCH_MODES="docker remote local"
CLIENT_PROTOCOLS="http grpc"
TRITON_DOCKER_IMAGE="nvcr.io/nvidia/tritonserver:21.03-py3"

# Run the model-analyzer, both client protocols
RET=0

function check_all_gpus() {
    analyzer_gpus=($@)
    gpu_uuids=(`get_all_gpus_uuids`)
    cuda_devices=(`echo $CUDA_VISIBLE_DEVICES | sed "s/,/ /g"`)

    if [ ${#analyzer_gpus[@]} != ${#cuda_devices[@]} ]; then
        return 1
    fi

    index=0
    for cuda_index in $cuda_devices; do
        gpu_uuid=${gpu_uuids[$cuda_index]}
        if [ $gpu_uuid != ${analyzer_gpus[$index]} ]; then
            echo -e "\n***\n*** Model Analyzer is not using the correct GPUs.\n***"
            return 1
        fi
        index=$((index+1))
    done
    return 0
}

function check_gpus() {
    analyzer_gpus=($1)
    triton_gpus=($2)

    if [ ${#analyzer_gpus[@]} != ${#triton_gpus[@]} ]; then
        return 1
    fi

    index=0
    for analyzer_gpu in $analyzer_gpus; do
        gpu_uuid=${triton_gpus[$index]}
        if [ $gpu_uuid != $analyzer_gpu ]; then
            echo -e "\n***\n*** Model Analyzer is not using the correct GPUs.\n***"
            return 1
        fi
        index=$((index+1))
    done
    return 0
}

function convert_gpu_array_to_flag() {
    gpu_array=$1
    if [ ${#gpu_array[@]} -ge 1 ]; then
        gpus_flag="--gpus "
        for gpu in ${gpu_array[@]}; do
            gpus_flag="$gpu_flag $gpu,"
        done

        # Remove trailing comma
        gpu_flag=${gpu_flag::-1}
        echo $gpu_flag
    else
        echo ""
    fi
}

function run_server_launch_modes() {
    gpus=$1
    for PROTOCOL in $CLIENT_PROTOCOLS; do
        MODEL_ANALYZER_ARGS_WITH_PROTOCOL="$MODEL_ANALYZER_BASE_ARGS --client-protocol=$PROTOCOL `convert_gpu_array_to_flag $gpus`"
        for LAUNCH_MODE in $TRITON_LAUNCH_MODES; do
            rm -rf $OUTPUT_MODEL_REPOSITORY && rm -rf $CHECKPOINT_DIRECTORY/*
            MODEL_ANALYZER_ARGS_WITH_LAUNCH_MODE="$MODEL_ANALYZER_ARGS_WITH_PROTOCOL --triton-launch-mode=$LAUNCH_MODE"
            ANALYZER_LOG=analyzer.${LAUNCH_MODE}.${PROTOCOL}.log
            SERVER_LOG=${LAUNCH_MODE}.${PROTOCOL}.server.log

            # Set arguments for various launch modes
            if [ "$LAUNCH_MODE" == "local" ]; then    
                MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS_WITH_LAUNCH_MODE $MODEL_ANALYZER_PORTS --triton-output-path=${SERVER_LOG}"
            elif [ "$LAUNCH_MODE" == "docker" ]; then
                MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS_WITH_LAUNCH_MODE $MODEL_ANALYZER_PORTS --triton-output-path=${SERVER_LOG} --triton-docker-image=$TRITON_DOCKER_IMAGE"
            elif [ "$LAUNCH_MODE" == "remote" ]; then
                MODEL_ANALYZER_PORTS="--triton-http-endpoint localhost:8000 --triton-grpc-endpoint localhost:8001"
                MODEL_ANALYZER_PORTS="$MODEL_ANALYZER_PORTS --triton-metrics-url http://localhost:8002/metrics"
                MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS_WITH_LAUNCH_MODE $MODEL_ANALYZER_PORTS"

                # For remote launch, set server args and start server
                SERVER=`which tritonserver`
                SERVER_ARGS="--model-repository=$MODEL_REPOSITORY --model-control-mode=explicit"
                
                run_server
                if [ "$SERVER_PID" == "0" ]; then
                    echo -e "\n***\n*** Failed to start $SERVER\n***"
                    cat $SERVER_LOG
                    exit 1
                fi
            fi

            # Run the analyzer and check the results
            set +e
            MODEL_ANALYZER_SUBCOMMAND="profile"
            run_analyzer
            if [ $? -ne 0 ]; then
                echo -e "\n***\n*** Test with launch mode '${LAUNCH_MODE}' using ${PROTOCOL} client Failed."\
                        "\n***     model-analyzer exited with non-zero exit code. \n***"
                cat $ANALYZER_LOG
                RET=1
            fi
            set -e

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

            model_analyzer_gpu_uuids=`cat $ANALYZER_LOG | grep -E "Using GPU\(s\) with UUID\(s\)" | tail -n 1 | sed -n 's/.*{\(.*\)}.*/\1/p'`
            model_analyzer_gpu_uuids=(`echo $model_analyzer_gpu_uuids | sed "s/,/ /g"`)
            if [ -z $gpus ]; then
                check_all_gpus "${model_analyzer_gpu_uuids[@]}"
            else
                check_gpus "${model_analyzer_gpu_uuids[@]}" "${gpus[@]}"
            fi
            if [ $? -ne 0 ]; then
                RET=1
                break
            fi
        done
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

#################################################
# Test controling the GPUs with the --gpus flag #
#################################################
CURRENT_GPUS=(${GPUS[2]})
run_server_launch_modes "$CURRENT_GPUS"

CURRENT_GPUS=(${GPUS[2]})
run_server_launch_modes "$CURRENT_GPUS"

CURRENT_GPUS=${GPUS[@]:1}
run_server_launch_modes "$CURRENT_GPUS"

CURRENT_GPUS=${GPUS[@]:1}
run_server_launch_modes "$CURRENT_GPUS"

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
