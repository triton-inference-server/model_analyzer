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

from .mock_dcgm_agent import TEST_PCI_BUS_ID
from .mock_base import MockBase


class MockNumba(MockBase):
    """
    Mocks numba class
    """

    def _fill_patchers(self):
        patchers = self._patchers

        numba_imports_path = ['model_analyzer.device.gpu_device_factory']

        for import_path in numba_imports_path:
            device = MagicMock()

            # Ignore everything after 0
            test_pci_id = str(TEST_PCI_BUS_ID, encoding='ascii').split('.')[0]

            pci_domain_id, pci_bus_id, pci_device_id = test_pci_id.split(':')
            device.get_device_identity = MagicMock(return_value={
                pci_bus_id: int(pci_bus_id, 16),
                pci_domain_id: int(pci_domain_id, 16),
                pci_device_id: int(pci_device_id, 16)
            })
            patchers.append(
                patch.multiple(f'{import_path}.numba.cuda',
                               list_devices=[device]))
