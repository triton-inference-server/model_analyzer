#!/bin/bash
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


###############################
# Test that if we run model_analyzer on a model with no config.pbtxt,
# that we can generate a default model config (in some cases)
#
# We test this by copying some models locally into a temp model repository
# and deleting the config files before running
#
# Only local and docker modes are tested and are expected to work
# - Remote mode is excluded because the model must already be loaded before
#   calling model analyzer in remote mode
# - c_api mode is excluded because perf_analyzer is our client for c_api mode
#   and does not support this functionality
###############################

source ../common/util.sh
create_logs_dir "L0_no_config_file"

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}

OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
rm -rf $OUTPUT_MODEL_REPOSITORY *.log

MODEL_REPOSITORY=$(get_output_directory)
rm -rf $MODEL_REPOSITORY
mkdir -p $MODEL_REPOSITORY
cp -R /mnt/nvdl/datasets/inferenceserver/$REPO_VERSION/qa_model_repository/onnx_int32_int32_int32 $MODEL_REPOSITORY
cp -R /mnt/nvdl/datasets/inferenceserver/$REPO_VERSION/libtorch_model_store/vgg19_libtorch $MODEL_REPOSITORY
rm $MODEL_REPOSITORY/*/config.pbtxt

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
TRITON_DOCKER_IMAGE=${TRITONSERVER_BASE_IMAGE_NAME}
TRITON_LAUNCH_MODES="local docker"
CLIENT_PROTOCOLS="http grpc"
PORTS=(`find_available_ports 3`)
http_port="${PORTS[0]}"
grpc_port="${PORTS[1]}"
metrics_port="${PORTS[2]}"
MODEL_ANALYZER_BASE_ARGS="-m $MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-http-endpoint localhost:$http_port --triton-grpc-endpoint localhost:$grpc_port"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-metrics-url http://localhost:$metrics_port/metrics"

python3 test_config_generator.py --protocols "`echo $CLIENT_PROTOCOLS | sed 's/ /,/g'`" --launch-modes "`echo $TRITON_LAUNCH_MODES | sed 's/ /,/g'`"

LIST_OF_CONFIG_FILES=(`ls | grep .yaml`)

if [ ${#LIST_OF_CONFIG_FILES[@]} -le 0 ]; then
    echo -e "\n***\n*** Test Failed. No config file exists. \n***"
    exit 1
fi

RET=0

function run_models() {
    # ONNX model should be able to create a default config
    PROFILE_MODEL="onnx_int32_int32_int32"
    MA_EXPECTED_RESULT="EP"
    run_server_launch_modes

    # Libtorch model should NOT be able to create a default config
    PROFILE_MODEL="vgg19_libtorch"
    MA_EXPECTED_RESULT="EF"
    run_server_launch_modes
}

function run_server_launch_modes() {
    for CONFIG_FILE in ${LIST_OF_CONFIG_FILES[@]}; do
        # e.g. config-docker-http.yaml -> config-docker-http
        CONFIG_PARAMETERS=${CONFIG_FILE%".yaml"}

        # config-docker-http -> [config, docker, http]
        PARAMETERS=(${CONFIG_PARAMETERS//-/ })
        LAUNCH_MODE=${PARAMETERS[1]}
        PROTOCOL=${PARAMETERS[2]}

        TEST_NAME=${PROFILE_MODEL}.${LAUNCH_MODE}.${PROTOCOL}
        create_result_paths -test-name $TEST_NAME

        SERVER_LOG=$TEST_LOG_DIR/server.${TEST_NAME}.log

        MODEL_ANALYZER_GLOBAL_OPTIONS="-v"
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS -f $CONFIG_FILE --profile-models $PROFILE_MODEL --checkpoint-directory $CHECKPOINT_DIRECTORY -e $EXPORT_PATH"

        _run_single_config

        if [ $? -ne 0 ]; then
            exit 1
        fi
    done
}

function _run_single_config() {
    if [ "$LAUNCH_MODE" == "docker" ]; then
        # Login to the GitLab CI. Required for pulling the Triton container.
        docker login -u gitlab-ci-token -p "${CI_JOB_TOKEN}" "${CI_REGISTRY}"

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

    if [ ! -s "$SERVER_LOG" ]; then
        echo -e "\n***\n*** Test Output Verification Failed : No logs found\n***"
        cat $ANALYZER_LOG
        RET=1
        exit 1
    fi

    if [ $? -ne 0 ]; then
        RET=1
        return 1
    fi
    set -e
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


run_models

rm -rf $MODEL_REPOSITORY *.yaml

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
