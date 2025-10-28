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

source ../common/util.sh
create_logs_dir "L0_profile_quick"

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/nvdl/datasets/inferenceserver/$REPO_VERSION/libtorch_model_store"}
TRITON_LAUNCH_MODE=${TRITON_LAUNCH_MODE:="local"}
CLIENT_PROTOCOL="http"
PORTS=(`find_available_ports 3`)
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}

create_result_paths
rm -rf $OUTPUT_MODEL_REPOSITORY

# Run the analyzer and check the results
RET=0

set +e

MODEL_ANALYZER_ARGS="-m $MODEL_REPOSITORY --profile-models vgg19_libtorch --run-config-search-mode=quick"

MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-http-endpoint localhost:${PORTS[0]} --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-metrics-url http://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS -e $EXPORT_PATH --checkpoint-directory $CHECKPOINT_DIRECTORY"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --run-config-search-max-concurrency 2 --run-config-search-max-instance-count 2"
MODEL_ANALYZER_SUBCOMMAND="profile"

run_analyzer

if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
