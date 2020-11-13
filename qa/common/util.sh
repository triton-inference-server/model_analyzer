# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
