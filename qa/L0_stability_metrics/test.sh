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

rm -f *.log

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/model_analyzer_benchmark_models"}
CHECKPOINT_REPOSITORY=${CHECKPOINT_REPOSITORY:="/mnt/dldata/inferenceserver/model_analyzer_checkpoints"}
TRITON_LAUNCH_MODE=${TRITON_LAUNCH_MODE:="local"}
CLIENT_PROTOCOL="grpc"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
CONFIG_FILE="config.yaml"
NUM_ITERATIONS=${NUM_ITERATIONS:=5}
BENCHMARK_MODELS="`ls ${MODEL_REPOSITORY}`"
MODEL_NAMES="$(echo $BENCHMARK_MODELS | sed 's/ /,/g')"
EXPORT_PATH="`pwd`/results"
FILENAME_SERVER_ONLY="server-metrics.csv"
FILENAME_INFERENCE_MODEL="model-metrics-inference.csv"
FILENAME_GPU_MODEL="model-metrics-gpu.csv"
CHECKPOINT_DIRECTORY="./checkpoints"
CSV_PATH='.'

# Set up checkpoints
mkdir $EXPORT_PATH

# Generate test configs
python3 test_config_generator.py --models $MODEL_NAMES

# Set analyzer config options
RET=0

set +e

# Run the analyzer and check the results
for (( i=1; i<=$NUM_ITERATIONS; i++ )); do
    # First profile
    MODEL_ANALYZER_ARGS="-m $MODEL_REPOSITORY -f $CONFIG_FILE"
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --gpus 0"
    MODEL_ANALYZER_SUBCOMMAND="profile"
    
    run_analyzer
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
        cat $ANALYZER_LOG
        RET=1
    fi

    # Then generate results
    MODEL_ANALYZER_ARGS="-e $EXPORT_PATH -f $CONFIG_FILE --filename-server-only=$FILENAME_SERVER_ONLY"
    MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --filename-model-inference=$FILENAME_INFERENCE_MODEL --filename-model-gpu=$FILENAME_GPU_MODEL "
    MODEL_ANALYZER_SUBCOMMAND="analyze"
    
    run_analyzer
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
        cat $ANALYZER_LOG
        RET=1
    fi
    mv $EXPORT_PATH/results/$FILENAME_GPU_MODEL $CSV_PATH/result_${i}.csv
    rm $CHECKPOINT_DIRECTORY/*
done

# Check the Analyzer log for correct output
TEST_NAME='metrics_stability'
python3 check_results.py -f $CONFIG_FILE -t $TEST_NAME -r $CSV_PATH
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Output Verification Failed for $TEST_NAME test.\n***"
    cat $ANALYZER_LOG
    RET=1
fi
set -e

rm -rf $EXPORT_PATH
rm -rf $OUTPUT_MODEL_REPOSITORY
rm -rf $CHECKPOINT_DIRECTORY
rm *.csv

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
