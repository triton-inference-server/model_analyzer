#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import total_ordering

from model_analyzer.record.gpu_record import IncreasingGPURecord


@total_ordering
class GPUFreeMemory(IncreasingGPURecord):
    """
    The free memory in the GPU.
    """

    tag = "gpu_free_memory"

    @staticmethod
    def value_function():
        """
        Returns the total value from a list

        Returns
        -------
        Total value of the list
        """
        return sum

    def __init__(self, value, device_uuid=None, timestamp=0):
        """
        Parameters
        ----------
        value : float
            The value of the GPU metrtic
        device_uuid : str
            The  GPU device uuid this metric is associated
            with.
        timestamp : int
            The timestamp for the record in nanoseconds
        """

        super().__init__(value, device_uuid, timestamp)

    @staticmethod
    def header(aggregation_tag=False):
        """
        Parameters
        ----------
        aggregation_tag: bool
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

        return ("Max " if aggregation_tag else "") + "GPU Memory Available (MB)"

    def __eq__(self, other):
        """
        Allows checking for
        equality between two records
        """

        return self.value() == other.value()

    def __lt__(self, other):
        """
        Allows checking if
        this record is less than
        the other
        """

        return self.value() < other.value()

    def __add__(self, other):
        """
        Allows adding two records together
        to produce a brand new record.
        """

        return GPUFreeMemory(device_uuid=None, value=(self.value() + other.value()))

    def __sub__(self, other):
        """
        Allows subtracting two records together
        to produce a brand new record.
        """

        return GPUFreeMemory(device_uuid=None, value=(self.value() - other.value()))
