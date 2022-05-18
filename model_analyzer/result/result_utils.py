# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
"""
A module for various functions
needed across the result module
"""


def format_for_csv(obj, interior=False):
    """
    Takes an object, which could be a string, int, or list of either
    and formats it so it will be parsable in a csv 
    """
    if type(obj) == list:
        if len(obj) > 1:
            if interior:
                return f" [{','.join([str(o) for o in obj])}]"
            else:
                return "\"" + ",".join(
                    [format_for_csv(o, interior=True) for o in obj]) + "\""
        else:
            return str(obj[0])
    elif type(obj) == str and "," in obj:
        return "\"" + obj + "\""
    else:
        return str(obj)
