# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

TEST_LOG='test.log'
EXPECTED_NUM_TESTS=`python3 count_tests.py --path ../../tests/`
source ../common/check_analyzer_results.sh

RET=0

set +e
python3 -m unittest discover -v -s ../../tests  -t ../../ > $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    RET=1
else
    check_unit_test_results $TEST_LOG $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $TEST_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    cat $TEST_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
