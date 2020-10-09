#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from model_analyzer.monitor.record import Record


class RecordCollector:

    def __init__(self):
        self._records = []

    def insert(self, record):
        """Insert a record into the RecordCollector

        Parameters
        ----------
        record : Record
            A record to be inserted
        """
        if isinstance(record, Record):
            self._records.append(record)

    def get(self, index):
        """Get record in the index

        Parameters
        ----------
        index : int
            index of the record to be returned

        Returns
        -------
        Record
            Record at location index
        """
        return self._records[index]

    def size(self):
        """Get size of this RecordCollector

        Returns
        -------
        int
            Size of the RecordCollector
        """
        return len(self._records)
