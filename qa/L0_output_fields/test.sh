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
python3 config_generator.py

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/$REPO_VERSION/libtorch_model_store"}
CHECKPOINT_REPOSITORY=${CHECKPOINT_REPOSITORY:="/mnt/dldata/inferenceserver/model_analyzer_checkpoints"}
EXPORT_PATH="`pwd`/results"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CHECKPOINT_DIRECTORY="."

cp $CHECKPOINT_REPOSITORY/export_metrics.ckpt $CHECKPOINT_DIRECTORY/0.ckpt

MODEL_ANALYZER_ANALYZE_BASE_ARGS="-e $EXPORT_PATH --checkpoint-directory $CHECKPOINT_DIRECTORY --summarize=False --filename-server-only=$FILENAME_SERVER_ONLY"
MODEL_ANALYZER_ANALYZE_BASE_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"
MODEL_ANALYZER_SUBCOMMAND="analyze"
LIST_OF_CONFIG_FILES=(`ls | grep .yml`)

RET=0

if [ ${#LIST_OF_CONFIG_FILES[@]} -le 0 ]; then
    echo -e "\n***\n*** Test Failed. No config file exists. \n***"
    RET=1
    exit $RET
fi

# Run the analyzer with various configurations and check the results
RET=0

for CONFIG_FILE in ${LIST_OF_CONFIG_FILES[@]}; do
    rm -rf results && mkdir -p results
    set +e

    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS -f $CONFIG_FILE" 
    ANALYZER_LOG=analyzer.${CONFIG_FILE}.log

    TEST_OUTPUT_NUM_ROWS=1
    run_analyzer
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed. model-analyzer exited with non-zero exit code. \n***"
        cat $ANALYZER_LOG
        RET=1
    else
        SERVER_METRICS_FILE=${EXPORT_PATH}/results/${FILENAME_SERVER_ONLY}
        MODEL_METRICS_GPU_FILE=${EXPORT_PATH}/results/${FILENAME_GPU_MODEL}
        MODEL_METRICS_INFERENCE_FILE=${EXPORT_PATH}/results/${FILENAME_INFERENCE_MODEL}

        METRICS_NUM_COLUMNS_FILE=`echo $CONFIG_FILE | sed 's/\.yml//'`-param-gpu.txt
        SERVER_NUM_COLUMNS_FILE=`echo $CONFIG_FILE | sed 's/\.yml//'`-param-server.txt
        INFERENCE_NUM_COLUMNS_FILE=`echo $CONFIG_FILE | sed 's/\.yml//'`-param-inference.txt
        METRICS_NUM_COLUMNS=`cat $METRICS_NUM_COLUMNS_FILE`
        INFERENCE_NUM_COLUMNS=`cat $INFERENCE_NUM_COLUMNS_FILE`
        SERVER_METRICS_NUM_COLUMNS=`cat $SERVER_NUM_COLUMNS_FILE`

        check_table_row_column \
            $ANALYZER_LOG $ANALYZER_LOG $ANALYZER_LOG \
            $MODEL_METRICS_INFERENCE_FILE $MODEL_METRICS_GPU_FILE $SERVER_METRICS_FILE \
            $INFERENCE_NUM_COLUMNS $TEST_OUTPUT_NUM_ROWS \
            $METRICS_NUM_COLUMNS $TEST_OUTPUT_NUM_ROWS \
            $SERVER_METRICS_NUM_COLUMNS 1
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

rm -f *.ckpt

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
