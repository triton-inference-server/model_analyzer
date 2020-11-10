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

from unittest.mock import patch, Mock, MagicMock

from .mock_dcgm_field_group_watcher import MockDCGMFieldGroupWatcherHelper
from .mock_dcgm_agent import MockDCGMAgent


class MockDCGM:
    """
    Mocks dcgm_agent methods.
    """

    def __init__(self):
        self._patchers = patchers = []

        structs_imports_path = [
            'model_analyzer.monitor.dcgm.dcgm_monitor',
            'model_analyzer.device.gpu_device_factory'
        ]
        for import_path in structs_imports_path:
            patchers.append(
                patch(f'{import_path}.structs._dcgmInit', MagicMock()))

        dcgm_agent_imports_path = [
            'model_analyzer.monitor.dcgm.dcgm_monitor',
            'model_analyzer.device.gpu_device_factory'
        ]
        for import_path in dcgm_agent_imports_path:
            patchers.append(patch(f'{import_path}.dcgm_agent', MockDCGMAgent))

        patchers.append(
            patch(
                'model_analyzer.monitor.dcgm.dcgm_monitor.dcgm_field_helpers.DcgmFieldGroupWatcher',
                MockDCGMFieldGroupWatcherHelper,
            ))

    def start(self):
        for patch in self._patchers:
            patch.start()

    def stop(self):
        for patch in self._patchers:
            patch.stop()
