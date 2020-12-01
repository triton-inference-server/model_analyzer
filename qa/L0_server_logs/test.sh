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
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/dldata/inferenceserver/model_analyzer_clara_pipelines"}
MODEL_NAMES="classification_chestxray_v1"
BATCH_SIZES="4"
CONCURRENCY="4"
MODEL_ANALYZER_BASE_ARGS="-m $MODEL_REPOSITORY -n $MODEL_NAMES -b $BATCH_SIZES -c $CONCURRENCY"
TRITON_LAUNCH_MODES="docker local"
TRITON_SERVER_VERSION="20.11-py3"

# Run the model-analyzer, both client protocols
RET=0

for LAUNCH_MODE in $TRITON_LAUNCH_MODES; do
    ANALYZER_LOG=analyzer.${LAUNCH_MODE}.log
    SERVER_LOG=server.${LAUNCH_MODE}.log
    MODEL_ANALYZER_ARGS_WITH_LAUNCH_MODE="$MODEL_ANALYZER_BASE_ARGS --triton-launch-mode=$LAUNCH_MODE --triton-output-path=${SERVER_LOG}"
   
    # Set arguments for various launch modes
    if [ "$LAUNCH_MODE" == "local" ]; then    
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS_WITH_LAUNCH_MODE"
    elif [ "$LAUNCH_MODE" == "docker" ]; then
        MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS_WITH_LAUNCH_MODE --triton-version=$TRITON_SERVER_VERSION"
    fi

    # Run the analyzer and check the results
    set +e
    run_analyzer
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test with launch mode '${LAUNCH_MODE}' Failed."\
                "\n***     model-analyzer exited with non-zero exit code. \n***"
        cat $ANALYZER_LOG
        RET=1
    else
        if [ ! -s "$SERVER_LOG" ]; then
            echo -e "\n***\n*** Test Output Verification Failed : No logs found\n***"
            cat $ANALYZER_LOG
            RET=1
        fi
    fi
    set -e

    if [ "$LAUNCH_MODE" == "docker" ]; then
        python3 ../common/cleanup.py $CONTAINER_NAME || true
    fi
done


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
