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


class CoordinateData:
    """
    Holds throughputs and visit counts for coordinates
    """

    def __init__(self):
        self._throughputs = {}
        self._visit_counts = {}

    def get_throughput(self, coordinate):
        """
        Get the throughput for the given coordinate. 
        Returns None if no throughput has been stored for that coordinate yet
        """
        key = tuple(coordinate)
        return self._throughputs.get(key, None)

    def set_throughput(self, coordinate, throughput):
        """
        Set the throughput for the given coordinate. 
        """
        key = tuple(coordinate)
        self._throughputs[key] = throughput

    def get_visit_count(self, coordinate):
        """
        Get the visit count for the given coordinate. 
        Returns 0 if the coordinate hasn't been visited yet
        """
        key = tuple(coordinate)
        return self._visit_counts.get(key, 0)

    def increment_visit_count(self, coordinate):
        """
        Increase the visit count for the given coordinate by 1
        """
        key = tuple(coordinate)
        new_count = self.get_visit_count(coordinate) + 1
        self._visit_counts[key] = new_count
