# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
LOGS_DIR="/logs/L0_ssl_https"

apt update ; apt install -y nginx

mkdir -p /tmp/output

# Set test parameters
MODEL_ANALYZER="`which model-analyzer`"
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
MODEL_REPOSITORY=${MODEL_REPOSITORY:="/mnt/nvdl/datasets/inferenceserver/$REPO_VERSION/libtorch_model_store"}
TRITON_LAUNCH_MODE=${TRITON_LAUNCH_MODE:="remote"}
CLIENT_PROTOCOL="http"
PORTS=(`find_available_ports 2`)
HTTP_PORT="8000"
GPUS=(`get_all_gpus_uuids`)
OUTPUT_MODEL_REPOSITORY=${OUTPUT_MODEL_REPOSITORY:=`get_output_directory`}
WORKING_CONFIG_FILE="working_config.yml"
BROKEN_CONFIG_FILE="broken_config.yml"

rm -rf $OUTPUT_MODEL_REPOSITORY

# Run the analyzer and check the results
RET=0

# Generate valid CA
openssl genrsa -passout pass:1234 -des3 -out ca.key 4096
openssl req -passin pass:1234 -new -x509 -days 365 -key ca.key -out ca.crt -subj  "/C=SP/ST=Spain/L=Valdepenias/O=Test/OU=Test/CN=Root CA"

# Generate valid Server Key/Cert
openssl genrsa -passout pass:1234 -des3 -out server.key 4096
openssl req -passin pass:1234 -new -key server.key -out server.csr -subj  "/C=SP/ST=Spain/L=Valdepenias/O=Test/OU=Server/CN=localhost"
openssl x509 -req -passin pass:1234 -days 365 -in server.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out server.crt

# Remove passphrase from the Server Key
openssl rsa -passin pass:1234 -in server.key -out server.key

# Generate valid Client Key/Cert
openssl genrsa -passout pass:1234 -des3 -out client.key 4096
openssl req -passin pass:1234 -new -key client.key -out client.csr -subj  "/C=SP/ST=Spain/L=Valdepenias/O=Test/OU=Client/CN=localhost"
openssl x509 -passin pass:1234 -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out client.crt

# Remove passphrase from Client Key
openssl rsa -passin pass:1234 -in client.key -out client.key

cp server.crt /etc/nginx/cert.crt
cp server.key /etc/nginx/cert.key

# Create mutated client key (Make first char of each like capital)
cp client.key client2.key && sed -i "s/\b\(.\)/\u\1/g" client2.key
cp client.crt client2.crt && sed -i "s/\b\(.\)/\u\1/g" client2.crt

# For remote launch, set server args and start server
SERVER=`which tritonserver`
SERVER_ARGS="--model-repository=$MODEL_REPOSITORY --model-control-mode explicit --http-port ${HTTP_PORT} --grpc-port ${PORTS[0]} --metrics-port ${PORTS[1]}"
SERVER_HTTP_PORT=${HTTP_PORT}
SERVER_LOG="$LOGS_DIR/server.log"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

cp nginx.conf /etc/nginx/sites-available/default

service nginx restart

set +e

# Test with working keys
TEST_NAME="test_working_keys"
TEST_LOG_DIR="$LOGS_DIR/$TEST_NAME/logs"
ANALYZER_LOG="$TEST_LOG_DIR/test.log"
EXPORT_PATH="$LOGS_DIR/$TEST_NAME/results"
mkdir -p $TEST_LOG_DIR $EXPORT_PATH

MODEL_ANALYZER_ARGS="-m $MODEL_REPOSITORY -f $WORKING_CONFIG_FILE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-http-endpoint https://localhost:443 --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-metrics-url https://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_SUBCOMMAND="profile"
run_analyzer
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed. model-analyzer $MODEL_ANALYZER_SUBCOMMAND exited with non-zero exit code. \n***"
    cat $ANALYZER_LOG
    RET=1
fi

# Test with broken keys
TEST_NAME="test_broken_keys"
TEST_LOG_DIR="$LOGS_DIR/$TEST_NAME/logs"
ANALYZER_LOG="$TEST_LOG_DIR/test.log"
EXPORT_PATH="$LOGS_DIR/$TEST_NAME/results"
mkdir -p $TEST_LOG_DIR $EXPORT_PATH

MODEL_ANALYZER_ARGS="-m $MODEL_REPOSITORY -f $BROKEN_CONFIG_FILE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --client-protocol=$CLIENT_PROTOCOL --triton-launch-mode=$TRITON_LAUNCH_MODE"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-http-endpoint https://localhost:443 --triton-grpc-endpoint localhost:${PORTS[1]}"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --triton-metrics-url https://localhost:${PORTS[2]}/metrics"
MODEL_ANALYZER_ARGS="$MODEL_ANALYZER_ARGS --output-model-repository-path $OUTPUT_MODEL_REPOSITORY --override-output-model-repository"
MODEL_ANALYZER_SUBCOMMAND="profile"
run_analyzer
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Expected Test Failure. \n***"
    cat $ANALYZER_LOG
    RET=1
fi

set -e

service nginx stop

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
