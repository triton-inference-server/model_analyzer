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

RET=0

model-analyzer profile \
  -m /quick_start_repository/ \
  --profile-models add_sub \
  --run-config-search-max-concurrency 2 \
  --run-config-search-max-instance-count 2 \
  --run-config-search-preferred-batch-size-disable true ; \
  if [ $? -ne 0 ] ; then RET=1 ; fi

mkdir analysis_results ; \
  if [ $? -ne 0 ] ; then RET=1 ; fi

model-analyzer analyze \
  --analysis-models add_sub \
  -e analysis_results ; \
  if [ $? -ne 0 ] ; then RET=1 ; fi

model-analyzer report \
  --report-model-configs add_sub_i0,add_sub_i1 \
  -e analysis_results ; \
  if [ $? -ne 0 ] ; then RET=1 ; fi

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test PASSED\n***"
  exit 0
else
  echo -e "\n***\n*** Test FAILED\n***"
  exit 1
fi
