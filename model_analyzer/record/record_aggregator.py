# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from model_analyzer.record.record import Record
from collections import defaultdict


class RecordAggregator:
    """
    Stores a collection of Record objects.
    """

    def __init__(self):
        self._records = defaultdict(list)

    def insert(self, record):
        """
        Insert a record into the RecordAggregator

        Parameters
        ----------
        record : Record
            A record to be inserted
        """

        if isinstance(record, Record):
            record_type = type(record)
            self._records[record_type].append(record)
        else:
            raise TritonModelAnalyzerException(
                "Can only add objects of type 'Record' to RecordAggregator")

    def filter_records(self, record_types=None, filters=None):
        """
        Get records that satisfy
        the given list of criteria.

        Parameters
        ----------

        record_types : list of types of Records
            the types of the records we are
            imposing the filter criteria on.

        filters : list of callables
            conditions that determine whether
            a given record should be returned.
            If no filters specified, all records
            of types specified by record_types will be
            returned.
            Note : This must be of the same length
                   as the list of record_types, or omitted.

        Returns
        -------
        Dict of List of Records
            keys are record_types, values
            are all records of that type
        """

        if not record_types and not filters:
            return self._records
        if record_types and not filters:
            try:
                return {k: self._records[k] for k in record_types}
            except KeyError as k:
                raise TritonModelAnalyzerException(
                    f"Record type '{k.header()}' not found in this RecordAggregator"
                )
        if filters and not record_types:
            raise TritonModelAnalyzerException(
                "Must specify the record types corresponding to each filter criterion."
            )
        if len(record_types) != len(filters):
            raise TritonModelAnalyzerException(
                "Must specify the same number of record types as filter criteria."
            )

        # Remove records that do not satisfy criteria
        filtered_records = defaultdict(list)
        for h, f in zip(record_types, filters):
            filtered_records[h] = [
                record for record in self._records[h] if f(record.value())
            ]

        return filtered_records

    def record_types(self):
        """
        Returns
        -------
        list of str
            a list of the types of records in this
            RecordAgrregator
        """

        return list(self._records.keys())

    def total(self, record_type=None):
        """
        Get the total number of records in
        the RecordAggregator

        Parameters
        ----------
        record_type : a class name of type Record
            The type of records to count,
            if None, count all types

        Returns
        -------
        int
            number of records in
            the RecordAggregator
        """

        if record_type:
            if record_type not in self._records:
                raise TritonModelAnalyzerException(
                    f"Record type '{record_type.header()}' not found in this RecordAggregator"
                )
            return len(self._records[record_type])
        return sum(len(self._records[k]) for k in self._records)

    def aggregate(self, record_types=None, reduce_func=max):
        """
        Parameters
        ----------
        record_types : List of Record types
            The type of records to aggregate.
            If None, aggregates all records

        reduce_func : callable
            takes as input a list of values
            and returns one

        Returns
        -------
        dict
            keys are requested record types
            and values are the aggregated values
        """

        aggregated_records = {}
        if not record_types:
            record_types = self.record_types()
        for record_type in record_types:
            values = []
            for record in self._records[record_type]:
                values.append(record.value())
            aggregated_records[record_type] = reduce_func(values)
        return aggregated_records
