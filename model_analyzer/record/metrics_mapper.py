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

from .record import RecordType


class MetricsMapper:
    """
    Class maps metrics to record types
    """

    @staticmethod
    def get_monitoring_metrics(tags):
        """
        Parameters
        ----------
        tags : list of str
            Human readable names for the 
            metrics to monitor. They correspond
            to actual record types.

        Returns
        -------
        List
            of record types being monitored
        """

        return [RecordType.get(tag) for tag in tags]
