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

from model_analyzer.device.device import Device


class GPUDevice(Device):
    """
    Representing a GPU device
    """

    def __init__(self, device_name, device_id, pci_bus_id, device_uuid):
        """
        Parameters
        ----------
            device_name: str
                Human readable name of the device
            device_id : int
                Device id according to the `nvidia-smi` output
            pci_bus_id : str
                PCI bus id
            device_uuid : str
                Device UUID
        """

        assert type(device_name) is str
        assert type(device_id) is int
        assert type(pci_bus_id) is str
        assert type(device_uuid) is str

        self._device_name = device_name
        self._device_id = device_id
        self._pci_bus_id = pci_bus_id
        self._device_uuid = device_uuid

    def device_name(self):
        """
        Returns
        -------
        str
            device name
        """

        return self._device_name

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
