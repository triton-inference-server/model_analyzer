# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

exit_early_if_nonzero () {
  if [[ $1 -ne 0 ]] ; then \
    echo -e "\n***\n*** Test FAILED\n***" ; \
    exit 1 ; \
  fi
}

model-analyzer profile \
  -m /opt/triton-model-analyzer/examples/quick-start \
  --profile-models add_sub \
  --run-config-search-max-concurrency 2 \
  --run-config-search-max-instance-count 2 ; \
  exit_early_if_nonzero $?

mkdir analysis_results ; \
  exit_early_if_nonzero $?

model-analyzer analyze \
  --analysis-models add_sub \
  -e analysis_results ; \
  exit_early_if_nonzero $?

model-analyzer report \
  --report-model-configs add_sub_config0,add_sub_config1 \
  -e analysis_results ; \
  exit_early_if_nonzero $?

echo -e "\n***\n*** Test PASSED\n***"
exit 0
