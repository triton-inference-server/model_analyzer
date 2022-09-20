# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

LOG="`pwd`/log.txt"
CONFIG="`pwd`/mkdocs.yml"
RET=0

#rm $LOG

exec mkdocs serve -f $CONFIG > $LOG &
PID=$!
sleep 20

until [[ (-z `pgrep mkdocs`) ]]; do
    kill -2 $PID
    sleep 2
done

if [[ ! -z `grep "invalid url" $LOG` ]]; then
    cat $LOG
    RET=1
fi


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi
exit $RET
