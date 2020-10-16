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
from model_analyzer.device.model import Device


class GPUDevice(Device):
    """
    Representing a GPU device
    """

    def __init__(self, device_id, pci_bus_id, device_uuid):
        """
        Parameters
        ----------
            device_id : int
                Device id according to the `nvidia-smi` output
            pci_bus_id : bytes
                PCI bus id
            device_uuid : str
                Device UUID
        """
        assert type(device_id) is int
        assert type(pci_bus_id) is bytes
        assert type(device_uuid) is str

        self._device_id = device_id
        self._pci_bus_id = pci_bus_id
        self._device_uuid = device_uuid

    def device_id(self):
        """
        Returns
        -------
        int
            device id of this GPU
        """
        return self._device_id

    def pci_bus_id(self):
        """
        Returns
        -------
        bytes
            PCI bus id of this GPU
        """
        return self._pci_bus_id

    def device_uuid(self):
        """
        Returns
        -------
        str
            UUID of this GPU
        """
        return self._device_uuid
