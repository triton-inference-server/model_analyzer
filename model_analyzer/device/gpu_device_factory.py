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
import numba.cuda
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByPciBusId, \
    nvmlDeviceGetUUID, nvmlDeviceGetPciInfo, nvmlDeviceGetIndex
from model_analyzer.device.gpu_device import GPUDevice


class GPUDeviceFactory:
    """
    Factory class for creating GPUDevices
    """

    @staticmethod
    def create_device_by_bus_id(bus_id):
        """
        Create a GPU device by using its bus ID.

        Parameters
        ----------
        bus_id : bytes
            Bus id corresponding to the GPU. The bus id should be created by
            converting the colon separated hex notation into a bytes type
            using ascii encoding. The bus id before conversion to bytes
            should look like "00:65:00".

        Returns
        -------
        Device
            The device associated with this bus id.
        """
        nvmlInit()
        handle = nvmlDeviceGetHandleByPciBusId(bus_id)
        device_uuid = nvmlDeviceGetUUID(handle)
        device_id = nvmlDeviceGetIndex(handle)
        nvmlShutdown()

        gpu_device = GPUDevice(device_id, bus_id, device_uuid)
        return gpu_device

    @staticmethod
    def create_device_by_cuda_index(index):
        """
        Create a GPU device using the CUDA index. This includes the index
        provided by CUDA visible devices.

        Parameters
        ----------
        index : int
            index of the device in the list of visible CUDA devices.

        Returns
        -------
        Device
            The device associated with the index provided.

        Raises
        ------
        IndexError
            If the index is out of bound.
        """
        devices = numba.cuda.list_devices()
        if index > len(devices) - 1:
            raise IndexError

        cuda_device = devices[index]
        device_identity = cuda_device.get_device_identity()
        device_bus_id = '{:02x}:{:02x}:{:02x}'.format(
            device_identity['pci_domain_id'],
            device_identity['pci_bus_id'],
            device_identity['pci_device_id'])

        return GPUDeviceFactory.create_device_by_bus_id(
            bytes(device_bus_id, encoding='ascii'))
