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

from abc import ABC, abstractmethod


class OutputWriter(ABC):
    """
    Interface that receives a table
    and writes the table to a file or stream.
    """

    @abstractmethod
    def write(self, out):
        """
        Writes the output to a file
        (stdout, .txt, .csv etc.)

        Parameters
        ----------
        out : str
            The string to be written out

        Raises
        ------
        TritonModelAnalyzerException
            If there is an error or exception while writing
            the output.
        """
