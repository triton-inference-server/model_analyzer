#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

source ../common/util.sh
source ../common/check_analyzer_results.sh
create_logs_dir "L0_ensemble_model"

# Set test parameters
MODEL_ANALYZER="$(which model-analyzer)"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/opt/triton-model-analyzer/examples/quick-start"}
QA_MODELS="ensemble_add_sub"
MODEL_NAMES="$(echo $QA_MODELS | sed 's/ /,/g')"
TRITON_LAUNCH_MODE=${TRITON_LAUNCH_MODE:="local"}
CLIENT_PROTOCOL="grpc"
PORTS=($(find_available_ports 3))
GPUS=($(get_all_gpus_uuids))
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=$(get_output_directory)}
CONFIG_FILE="config.yml"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"

mkdir -p $MODEL_REPOSITORY/ensemble_add_sub/1
rm -rf $OUTPUT_MODEL_REPOSITORY
create_result_paths
SERVER_LOG=$TEST_LOG_DIR/server.log

python3 test_config_generator.py --profile-models $MODEL_NAMES

# Run the analyzer and check the results
RET=0

set +e

MODEL_ANALYZER_ARGS="-m $MODEL_REPOSITORY -f $CONFIG_FILE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS -e $EXPORT_PATH --checkpoint-directory $CHECKPOINT_DIRECTORY --filename-server-only=$FILENAME_SERVER_ONLY"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --skip-detailed-reports --triton-output-path=$SERVER_LOG"
MODEL_ANALYZER_SUBCOMMAND="profile"

run_analyzer

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    # Check the Analyzer log for correct output
    TEST_NAME='profile_logs'
    python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -l $ANALYZER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi

    SERVER_METRICS_FILE=${EXPORT_PATH}/results/${FILENAME_SERVER_ONLY}
    MODEL_METRICS_GPU_FILE=${EXPORT_PATH}/results/${FILENAME_GPU_MODEL}
    MODEL_METRICS_INFERENCE_FILE=${EXPORT_PATH}/results/${FILENAME_INFERENCE_MODEL}

    for file in SERVER_METRICS_FILE, MODEL_METRICS_GPU_FILE, MODEL_METRICS_INFERENCE_FILE; do
        check_no_csv_exists $file
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed.\n***"
            cat $ANALYZER_LOG
            RET=1
        fi
    done
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** First Test PASSED\n***"
else
    echo -e "\n***\n*** First Test FAILED\n***"
fi

# Run second test: Ensemble composing model parameter ranges
echo -e "\n***\n*** Running ensemble composing model parameter ranges test\n***"

# Clean up for second test
rm -f $ANALYZER_LOG
rm -rf $CHECKPOINT_DIRECTORY
rm -rf $OUTPUT_MODEL_REPOSITORY
create_result_paths

CONFIG_FILE_RANGES="config_composing_ranges.yml"

set +e

MODEL_ANALYZER_ARGS="-m $MODEL_REPOSITORY -f $CONFIG_FILE_RANGES"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS -e $EXPORT_PATH --checkpoint-directory $CHECKPOINT_DIRECTORY --filename-server-only=$FILENAME_SERVER_ONLY"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --skip-detailed-reports --triton-output-path=$SERVER_LOG"
MODEL_ANALYZER_SUBCOMMAND="profile"

run_analyzer

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND (composing ranges) exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
else
    # Check the Analyzer log for correct output
    TEST_NAME='composing_model_ranges'
    python3 check_results.py -f $CONFIG_FILE_RANGES -t $TEST_NAME -l $ANALYZER_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
        cat $ANALYZER_LOG
        RET=1
    fi
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** All Tests PASSED\n***"
else
    echo -e "\n***\n*** Some Tests FAILED\n***"
fi

exit $RET
