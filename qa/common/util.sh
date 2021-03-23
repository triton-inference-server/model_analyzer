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

SERVER_LOG=${SERVER_LOG:="./server.log"}
SERVER_TIMEOUT=${SERVER_TIMEOUT:=120}
SERVER_LD_PRELOAD=${SERVER_LD_PRELOAD:=""}
ANALYZER_LOG=${ANALYZER_LOG:="./test.log"}

# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function wait_for_server_ready() {
    local spid="$1"; shift
    local wait_time_secs="${1:-30}"; shift

    WAIT_RET=0

    local wait_secs=$wait_time_secs
    until test $wait_secs -eq 0 ; do
        if ! kill -0 $spid; then
            echo "=== Server not running."
            WAIT_RET=1
            return
        fi

        sleep 1;

        set +e
        code=`curl -s -w %{http_code} localhost:8000/v2/health/ready`
        set -e
        if [ "$code" == "200" ]; then
            return
        fi

        ((wait_secs--));
    done

    echo "=== Timeout $wait_time_secs secs. Server not ready."
    WAIT_RET=1
}

# Run inference server. Return once server's health endpoint shows
# ready or timeout expires. Sets SERVER_PID to pid of SERVER, or 0 if
# error (including expired timeout)
function run_server () {
    SERVER_PID=0

    if [ -z "$SERVER" ]; then
        echo "=== SERVER must be defined"
        return
    fi

    if [ ! -f "$SERVER" ]; then
        echo "=== $SERVER does not exist"
        return
    fi

    if [ -z "$SERVER_LD_PRELOAD" ]; then
      echo "=== Running $SERVER $SERVER_ARGS"
    else
      echo "=== Running LD_PRELOAD=$SERVER_LD_PRELOAD $SERVER $SERVER_ARGS"
    fi

    LD_PRELOAD=$SERVER_LD_PRELOAD $SERVER $SERVER_ARGS > $SERVER_LOG 2>&1 &
    SERVER_PID=$!

    wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT
    if [ "$WAIT_RET" != "0" ]; then
        kill $SERVER_PID || true
        SERVER_PID=0
    fi
}

# Run model-analyzer with args to completion.
function run_analyzer() {
    if [[ -z "$MODEL_ANALYZER" ]]; then
        echo -e "=== model-analyzer executable not found"
        return 1
    fi

    if [ ! -f "$MODEL_ANALYZER" ]; then
        echo "=== $MODEL_ANALYZER does not exist"
        return 1
    fi

    echo -e "=== Running $MODEL_ANALYZER $MODEL_ANALYZER_ARGS"
    $MODEL_ANALYZER $MODEL_ANALYZER_ARGS >> $ANALYZER_LOG 2>&1
    return $?
}

# Check Python unittest results.
function check_test_results () {
    local log_file=$1
    local expected_num_tests=$2

    if [[ -z "$expected_num_tests" ]]; then
        echo "=== expected number of tests must be defined"
        return 1
    fi

    num_failures=`cat $log_file | grep -E ".*total.*errors.*failures.*" | tail -n 1 | jq .failures`
    num_tests=`cat $log_file | grep -E ".*total.*errors.*failures.*" | tail -n 1 | jq .total`
    num_errors=`cat $log_file | grep -E ".*total.*errors.*failures.*" | tail -n 1 | jq .errors`

    # Number regular expression
    re='^[0-9]+$'

    if [[ $? -ne 0 ]] || ! [[ $num_failures =~ $re ]] || ! [[ $num_tests =~ $re ]] || \
     ! [[ $num_errors =~ $re ]]; then
        cat $log_file
        echo -e "\n***\n*** Test Failed: unable to parse test results\n***" >> $log_file
        return 1
    fi
    if [[ $num_errors != "0" ]] || [[ $num_failures != "0" ]] || [[ $num_tests -ne $expected_num_tests ]]; then
        cat $log_file
        echo -e "\n***\n*** Test Failed: Expected $expected_num_tests test(s), $num_tests test(s) executed, $num_errors test(s) had error, and $num_failures test(s) failed. \n***" >> $log_file
        return 1
    fi

    return 0
}

# Check row and columns of csv file
function check_csv_table_row_column() {
    local csv_file=$1
    local expected_num_columns=$2
    local expected_num_rows=$3
    local tag=$4
    if [[ ! -f "$csv_file" ]]; then
        echo -e "\n***\n*** Test Failed: $csv_file does not exist\n***"
        return 1
    fi

    num_rows=`awk -v pattern=$tag '$0 ~ pattern {getline; i=0; while(getline) {i+=1}; print i}' $csv_file`
    if [[ "$num_rows" != "$expected_num_rows" ]]; then
        echo -e "\n***\n*** Test Failed: Expected $expected_num_rows rows in $csv_file, got ${num_rows}\n***"
        return 1
    fi
    for i in $( seq 1 $expected_num_rows ); do
        num_columns_found=`awk -v pattern=$tag -F ',' -v row="$i" '$0 ~ pattern {for (n=0; n<row;n++) {getline}; print NF}' $csv_file`
        if [[ "$num_columns_found" != "$expected_num_columns" ]]; then
            echo -e "\n***\n*** Test Failed: Expected $expected_num_columns columns in row $i, got ${num_columns_found}\n***"
            return 1
        fi
    done
    return 0
}

function get_number_of_rows_logfile() {
    local log_file=$1
    num_rows_found=`awk  "BEGIN{i=0} /$tag/{flag=1;getline;getline} /^$/{flag=0} flag {i+=1} END{print i}" $log_file`
    echo $number_row_found
}

function get_number_of_rows_csv() {
    local csv_file=$1
    num_rows=`awk -v pattern=$tag '$0 ~ pattern {getline; i=0; while(getline) {i+=1}; print i}' $csv_file`
    echo $num_rows
}

# Check the output tables from the model-analyzer
# This function simply ensures that there is a
# Server-only table, a model table, and that
# They have the specified dimensions
function check_log_table_row_column() {
    local log_file=$1
    local expected_num_columns=$2
    local expected_num_rows=$3
    local tag=$4

    # Check number of rows
    num_rows_found=`awk  "BEGIN{i=0} /$tag/{flag=1;getline;getline} /^$/{flag=0} flag {i+=1} END{print i}" $log_file`
    if [[ "$num_rows_found" != "$expected_num_rows" ]]; then
        echo -e "\n***\n*** Test Failed: Expected $expected_num_rows rows $log_file, got ${num_rows_found}\n***"
        return 1
    fi

    # Check models table
    for i in $( seq 1 $expected_num_rows ); do
        # Columns in ith row
        num_columns_found=`awk -v row="$i" "/$tag/{getline; for (n=0; n<row;n++) {getline}; print NF}" $log_file`
        if [[ "$num_columns_found" != "$expected_num_columns" ]]; then
            echo -e "\n***\n*** Test Failed: Expected $expected_num_columns columns in row $i of $tag, got ${num_columns_found}\n***"
            return 1
        fi
    done
    return 0
}

function install_netstat() {
    netstat > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        return
    else
        DEBIAN_FRONTEND=noninteractive
        apt-get update -qq > /dev/null 2>&1
        DEBIAN_FRONTEND=noninteractive
        apt-get install net-tools -y -qq > /dev/null 2>&1
    fi
}

function find_available_ports() {
    install_netstat
    # First argument is the number of ports
    num_port=$1

    export ports=()
    for i in `seq 1 $num_port`; do
        current_port=$((10000 + $RANDOM % 10000))
        while [ `netstat -ano tcp |& grep :$current_port > /dev/null 2>&1` ]; do
            current_port=$(echo "$port + 1" | bc)
        done
        ports=("${ports[@]}" "$current_port")
    done
    echo ${ports[@]}
}

# Get a space delimited list of gpus with their UUIDs.
function get_all_gpus_uuids() {
    nvidia-smi --query-gpu=gpu_uuid,pci.bus_id --format=csv,noheader | awk 'BEGIN { FS = "," }{print $1}'
}

# We need to randomize the output directory so that in case the CI containers
# end up on the same host, it does not create a conflict
function get_output_directory() {
    random_number=$((10000 + $RANDOM % 10000))
    while [ `ls /tmp/output/ | grep $random_number` ]; do
        random_number=$(echo "$random_number + 1" | bc)
    done

    echo /tmp/output/$random_number
}
