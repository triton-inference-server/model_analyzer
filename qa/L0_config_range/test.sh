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

rm -f *.log
rm -rf results && mkdir -p results
python3 config_generator.py

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
QA_MODELS="vgg19_libtorch resnet50_libtorch"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/$REPO_VERSION/libtorch_model_store"}
EXPORT_PATH="`pwd`/results"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
TRITON_LAUNCH_MODE="docker"
CLIENT_PROTOCOL="grpc"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}

MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --model-repository $MODEL_REPOSITORY"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS -e $EXPORT_PATH --filename-server-only=$FILENAME_SERVER_ONLY"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_BASE_ARGS="$MODEL_ANALYZER_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"

LIST_OF_CONFIG_FILES=(`ls | grep .yml`)

RET=0

if [ ${#LIST_OF_CONFIG_FILES[@]} -lt 0 ]; then
    echo -e "\n***\n*** Test Failed. No config file exists. \n***"
    RET=1
    exit $RET
fi

# Run the analyzer with various configurations and check the results
for config in ${LIST_OF_CONFIG_FILES[@]}; do
    rm -rf results && mkdir -p results && rm -rf $OUTPUT_MODEL_REPOSITORY
    set +e

    ANALYZER_LOG=analyzer.${config}.log
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_BASE_ARGS -f $config"
    NUM_ROW_OUTPUT_FILE=`echo $config | sed 's/\.yml/\.txt/'`
    TEST_OUTPUT_NUM_ROWS=`cat $NUM_ROW_OUTPUT_FILE`
    run_analyzer
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
        cat $ANALYZER_LOG
        RET=1
    else
        SERVER_METRICS_FILE=${EXPORT_PATH}/results/${FILENAME_SERVER_ONLY}
        MODEL_METRICS_GPU_FILE=${EXPORT_PATH}/results/${FILENAME_GPU_MODEL}
        MODEL_METRICS_INFERENCE_FILE=${EXPORT_PATH}/results/${FILENAME_INFERENCE_MODEL}
        METRICS_NUM_COLUMNS=11
        INFERENCE_NUM_COLUMNS=10
        SERVER_METRICS_NUM_COLUMNS=5


        OUTPUT_TAG="Model"
        check_csv_table_row_column $SERVER_METRICS_FILE $SERVER_METRICS_NUM_COLUMNS $((1 * ${#GPUS[@]})) $OUTPUT_TAG
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed for $SERVER_METRICS_FILE.\n***"
            cat $ANALYZER_LOG
            RET=1
        fi
        check_csv_table_row_column $MODEL_METRICS_GPU_FILE $METRICS_NUM_COLUMNS $(($TEST_OUTPUT_NUM_ROWS * ${#GPUS[@]})) $OUTPUT_TAG
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed for $MODEL_METRICS_GPU_FILE.\n***"
            cat $ANALYZER_LOG
            RET=1
        fi
        check_csv_table_row_column $MODEL_METRICS_INFERENCE_FILE $INFERENCE_NUM_COLUMNS $TEST_OUTPUT_NUM_ROWS $OUTPUT_TAG
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed for $MODEL_METRICS_INFERENCE_FILE.\n***"
            cat $ANALYZER_LOG
            RET=1
        fi
    fi

    rm $ANALYZER_LOG
    set -e
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
