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

from .mock_dcgm_agent import MockDCGMAgent
from model_analyzer.record.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.gpu_used_memory import GPUUsedMemory
from model_analyzer.monitor.dcgm.dcgm_monitor import DCGMMonitor

from collections import defaultdict
from unittest.mock import MagicMock

import time

TEST_RECORD_VALUE = 2.4


class MockDCGMFieldGroupWatcherHelper:
    """
    Mock of the DCGMFieldGroupWatcher class
    """

    def __init__(self, handle, group_id, field_group, operation_mode,
                 update_freq, max_keep_age, max_keep_samples, start_timestamp):
        """
        handle : dcgm_handle
            DCGM handle from dcgm_agent.dcgmInit()
        groupId : int
            a DCGM group ID returned from dcgm_agent.dcgmGroupCreate
        fieldGroup : int
            DcgmFieldGroup() instance to watch fields for
        operationMode : dcgm_structs.DCGM_OPERATION_MODE
            a dcgm_structs.DCGM_OPERATION_MODE_? constant for if the host
            engine is running in lock step or auto mode
        updateFreq : float
            how often to update each field in usec
        maxKeepAge : int
            how long DCGM should keep values for in seconds
        maxKeepSamples : int
            is the maximum number of samples DCGM should ever cache for each
            field
        startTimestamp : int
            a base timestamp we should start from when first reading
            values. This can be used to resume a previous instance of a
            DcgmFieldGroupWatcher by using its _nextSinceTimestamp. 0=start
            with all cached data
        """

        self._handle = handle
        self._group_id = group_id.value
        self._field_group = field_group
        self._operation_mode = operation_mode
        self._update_freq = update_freq
        self._max_keep_age = max_keep_age
        self._max_keep_samples = max_keep_samples
        self._start_timestamp = start_timestamp
        self.values = defaultdict(lambda: defaultdict(MagicMock))

    def GetMore(self):
        """
        This function perfoms a single iteration of monitoring
        """

        group_name = list(MockDCGMAgent.device_groups)[self._group_id]
        device_group = MockDCGMAgent.device_groups[group_name]
        field_group_name = list(MockDCGMAgent.field_groups)[self._field_group]

        for device in device_group:
            for field in MockDCGMAgent.field_groups[field_group_name]:

                # Sample Record
                record = MagicMock()
                record.value = TEST_RECORD_VALUE
                record.ts = int(time.time() * 1e6)
                if not isinstance(self.values[device][field].values, list):
                    self.values[device][field].values = [record]
                else:
                    self.values[device][field].values.append(record)
