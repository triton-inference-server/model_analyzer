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

import time

from .record import Record


class CPUAvailableRAM(Record):
    """
    The Available CPU memory
    """

    tag = "cpu_available_ram"

    def __init__(self, free_mem, timestamp=0):
        """
        Parameters
        ----------
        free_mem : float
            CPU free memory
        timestamp : int
            The timestamp for the record in nanoseconds
        """

        super().__init__(free_mem, timestamp)

    @staticmethod
    def header(aggregation_tag=None):
        """
        Parameters
        ----------
        aggregation_tag : str
            An optional tag that may be displayed 
            as part of the header indicating that 
            this record has been aggregated using 
            max, min or average etc. 
             
        Returns
        -------
        str
            The full name of the
            metric.
        """

        return aggregation_tag + "RAM Available(MB)"
