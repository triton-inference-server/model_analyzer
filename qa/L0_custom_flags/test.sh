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
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/$REPO_VERSION/libtorch_model_store"}
QA_MODELS="vgg19_libtorch resnet50_libtorch"
MODEL_NAMES="$(echo $QA_MODELS | sed 's/ /,/g')"
TRITON_LOG_BASE="triton.log"
BATCH_SIZES="1"
CONCURRENCY="1"
TRITON_LAUNCH_MODE="local"
CLIENT_PROTOCOL="grpc"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CHECKPOINT_DIRECTORY="`pwd`/checkpoints"

MODEL_ANALYZER_BASE_ARGS="-m $MODEL_REPOSITORY -b $BATCH_SIZES -c $CONCURRENCY"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --run-config-search-disable"
MODEL_ANALYZER_SUBCOMMAND="profile"

rm -rf $OUTPUT_MODEL_REPOSITORY

python3 test_config_generator.py -m $MODEL_NAMES

LIST_OF_CONFIG_FILES=(`ls | grep .yml`)

if [ ${#LIST_OF_CONFIG_FILES[@]} -le 0 ]; then
    echo -e "\n***\n*** Test Failed. No config file exists. \n***"
    exit 1
fi

RET=0

for CONFIG_FILE in ${LIST_OF_CONFIG_FILES[@]}; do
    ANALYZER_LOG="$CONFIG_FILE.log"
    set +e
    rm -f $CHECKPOINT_DIRECTORY/*
    
    # Run the analyzer and check the results
    TRITON_LOG_PREFIX=${CONFIG_FILE#"config-"}
    TRITON_LOG_PREFIX=${TRITON_LOG_PREFIX%".yml"}
    TRITON_LOG=${TRITON_LOG_PREFIX}.${TRITON_LOG_BASE}
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-output-path=${TRITON_LOG} -f $CONFIG_FILE"

    run_analyzer
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
        cat $ANALYZER_LOG
        RET=1
    else
        python3 check_results.py -f $CONFIG_FILE -m $MODEL_NAMES --analyzer-log-file $ANALYZER_LOG --triton-log-file $TRITON_LOG 
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed for $ANALYZER_LOG.\n***"
            cat $ANALYZER_LOG
            RET=1
        fi
    fi
    set -e
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
