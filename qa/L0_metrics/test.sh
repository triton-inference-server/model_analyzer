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

ANALYZER_LOG="test.log"
source ../common/util.sh
source ../common/check_analyzer_results.sh

rm -f *.log
rm -rf results && mkdir -p results

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/$REPO_VERSION/libtorch_model_store"}
MODEL_NAMES="vgg19_libtorch"
EXPORT_PATH="`pwd`/results"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
TRITON_LAUNCH_MODE="local"
CLIENT_PROTOCOL="grpc"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CHECKPOINT_DIRECTORY="`pwd`/checkpoints"
MODEL_ANALYZER_PROFILE_BASE_ARGS="--model-repository $MODEL_REPOSITORY --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --client-protocol=$CLIENT_PROTOCOL"
MODEL_ANALYZER_PROFILE_BASE_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"

MODEL_ANALYZER_ANALYZE_BASE_ARGS="--analysis-models $MODEL_NAMES -e $EXPORT_PATH --filename-server-only=$FILENAME_SERVER_ONLY --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_ANALYZE_BASE_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL"

python3 test_config_generator.py -m $MODEL_NAMES

LIST_OF_CONFIG_FILES=(`ls | grep .yml`)

RET=0

if [ ${#LIST_OF_CONFIG_FILES[@]} -le 0 ]; then
    echo -e "\n***\n*** Test Failed. No config file exists. \n***"
    exit 1
fi

# Run the analyzer with various configurations and check the results
for config in ${LIST_OF_CONFIG_FILES[@]}; do
    rm -rf results && mkdir -p results && rm -rf $OUTPUT_MODEL_REPOSITORY && rm -rf $CHECKPOINT_DIRECTORY/*
    set +e

    MODEL_ANALYZER_PORTS="--triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]} --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_PROFILE_BASE_ARGS -f $config --triton-launch-mode $TRITON_LAUNCH_MODE"
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS $MODEL_ANALYZER_PORTS"

    ANALYZER_LOG=analyzer.${config}.log

    MODEL_ANALYZER_SUBCOMMAND="profile"
    run_analyzer
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
        cat $ANALYZER_LOG
        RET=1
    else
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ANALYZE_BASE_ARGS -f $config"
        MODEL_ANALYZER_SUBCOMMAND="analyze"
        run_analyzer
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
            cat $ANALYZER_LOG
            RET=1
        else
            SERVER_METRICS_FILE=${EXPORT_PATH}/results/${FILENAME_SERVER_ONLY}
            MODEL_METRICS_GPU_FILE=${EXPORT_PATH}/results/${FILENAME_GPU_MODEL}
            MODEL_METRICS_INFERENCE_FILE=${EXPORT_PATH}/results/${FILENAME_INFERENCE_MODEL}

            METRICS_NUM_COLUMNS_FILE=`echo $config | sed 's/\.yml//'`-param-gpu.txt
            SERVER_NUM_COLUMNS_FILE=`echo $config | sed 's/\.yml//'`-param-server.txt
            INFERENCE_NUM_COLUMNS_FILE=`echo $config | sed 's/\.yml//'`-param-inference.txt
            METRICS_NUM_COLUMNS=`cat $METRICS_NUM_COLUMNS_FILE`
            INFERENCE_NUM_COLUMNS=`cat $INFERENCE_NUM_COLUMNS_FILE`
            SERVER_METRICS_NUM_COLUMNS=`cat $SERVER_NUM_COLUMNS_FILE`

            check_table_row_column \
                $ANALYZER_LOG $ANALYZER_LOG $ANALYZER_LOG \
                $MODEL_METRICS_INFERENCE_FILE $MODEL_METRICS_GPU_FILE $SERVER_METRICS_FILE \
                $INFERENCE_NUM_COLUMNS 1 \
                $METRICS_NUM_COLUMNS $((1 * ${#GPUS[@]})) \
                $SERVER_METRICS_NUM_COLUMNS $((1 * ${#GPUS[@]}))
            if [ $? -ne 0 ]; then
                echo -e "\n***\n*** Test Output Verification Failed.\n***"
                cat $ANALYZER_LOG
                RET=1
            fi
        fi
    fi

    rm $ANALYZER_LOG
    rm -rf $EXPORT_PATH/*
    set -e
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
