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
source ../common/util.sh

rm -f *.log

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
TRITON_DOCKER_IMAGE=${TRITONSERVER_BASE_IMAGE_NAME}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/nvdl/datasets/inferenceserver/${REPO_VERSION}/qa_custom_ops/libtorch_custom_ops"}
CLIENT_PROTOCOL="grpc"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CONFIG_FILE="config.yaml"
NUM_ITERATIONS=${NUM_ITERATIONS:=4}
MODEL_NAMES="libtorch_modulo"
CHECKPOINT_DIRECTORY="./checkpoints"
TRITON_LOG_BASE="triton.log"
WAIT_TIMEOUT=1200

# Generate test configs
python3 test_config_generator.py --profile-models $MODEL_NAMES --preload-path "/usr/lib/$(uname -m)-linux-gnu/libpython3.10.so.1:$MODEL_REPOSITORY/libtorch_modulo/custom_modulo.so" --library-path /opt/tritonserver/backends/pytorch:'$LD_LIBRARY_PATH'

LIST_OF_CONFIG_FILES=(`ls | grep .yaml`)

if [ ${#LIST_OF_CONFIG_FILES[@]} -le 0 ]; then
    echo -e "\n***\n*** Test Failed. No config file exists. \n***"
    exit 1
fi

MODEL_ANALYZER_BASE_ARGS="-m $MODEL_REPOSITORY"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL"
MODEL_ANALYZER_SUBCOMMAND="profile"

RET=0

# Login to the GitLab CI. Required for pulling the Triton container.
docker login -u gitlab-ci-token -p "${CI_JOB_TOKEN}" "${CI_REGISTRY}"

set +e
for CONFIG_FILE in ${LIST_OF_CONFIG_FILES[@]}; do
    # Loop 
    WAIT_TIME_SECS=$WAIT_TIMEOUT

    # Set up logs
    LOG_PREFIX=${CONFIG_FILE#"config-"}
    LOG_PREFIX=${LOG_PREFIX%".yaml"}
    TRITON_LOG=${LOG_PREFIX}.${TRITON_LOG_BASE}

    ANALYZER_LOG=${LOG_PREFIX}.${ANALYZER_LOG_BASE}
    touch $TRITON_LOG
    MODEL_ANALYZER_GLOBAL_OPTIONS="-v"

    # Run analyzer
    if [[ "$LOG_PREFIX" == "c_api" ]]; then    
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS -f $CONFIG_FILE --perf-output-path=$TRITON_LOG"
    elif [[ "$LOG_PREFIX" == "docker" ]]; then
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS -f $CONFIG_FILE --triton-docker-image $TRITON_SERVER_CONTAINER_IMAGE_NAME --triton-output-path=$TRITON_LOG"
    else
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS -f $CONFIG_FILE --triton-output-path=$TRITON_LOG"
    fi           
    run_analyzer

    if [[ ! -z `grep "symbol lookup error" $TRITON_LOG` ]]; then
        cat $ANALYZER_LOG
        echo -e "\n***\n*** Test Failed. Symbol lookup error. \n***"
        RET=1
    elif [[ ! -z `grep "Exception\|Traceback" $ANALYZER_LOG` ]]; then
        cat $ANALYZER_LOG
        echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
        RET=1
    # In the docker/local launch modes, triton log errors will be returned by triton client and printed in ANALYZER_LOG
    elif [[ ! -z `grep "This op may not exist or may not be currently supported" $ANALYZER_LOG` ]]; then
        cat $ANALYZER_LOG
        echo -e "\n***\n*** Test Failed. custom op not found. \n***"
        RET=1
    # In the C_API, Triton errors are displayed in the perf_analyzer log
    elif [[ ! -z `grep "This op may not exist or may not be currently supported" $TRITON_LOG` ]]; then
        cat $TRITON_LOG
        echo -e "\n***\n*** Test Failed. custom op not found. \n***"
        RET=1
    elif [[ ! -z `grep "Failed to load" $ANALYZER_LOG` ]]; then
        cat $ANALYZER_LOG
        echo -e "\n***\n*** Test Failed. Failed to load model. \n***"
        RET=1
    elif [[ ! -z `grep "Profile complete. Profiled 0 configurations" $ANALYZER_LOG` ]]; then
        cat $TRITON_LOG
        echo -e "\n***\n*** Test Failed. nothing was profiled. \n***"
        RET=1
    fi

    if [[ -z `grep "Profile complete" $ANALYZER_LOG` ]]; then
        cat $ANALYZER_LOG
        echo -e "\n***\n*** Test Failed. model-analyzer hangs. \n***"
        RET=1
    fi
    until [[ (-z `pgrep model-analyzer`) || ("`grep 'SIGINT' $ANALYZER_LOG | wc -l`" -gt "3") ]]; do
        kill -2 $ANALYZER_PID
        sleep 0.5
    done
    wait $ANALYZER_PID
    rm -f $CHECKPOINT_DIRECTORY/*
done
set -e

rm -rf *.yaml checkpoints plots reports results

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
