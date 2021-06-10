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

import numba.cuda
from model_analyzer.device.gpu_device import GPUDevice
import model_analyzer.monitor.dcgm.dcgm_agent as dcgm_agent
import model_analyzer.monitor.dcgm.dcgm_structs as structs
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

import logging


class GPUDeviceFactory:
    """
    Factory class for creating GPUDevices
    """
    @staticmethod
    def create_device_by_bus_id(bus_id, dcgmPath=None):
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

        structs._dcgmInit(dcgmPath)
        dcgm_agent.dcgmInit()

        # Start DCGM in the embedded mode to use the shared library
        dcgm_handle = dcgm_agent.dcgmStartEmbedded(
            structs.DCGM_OPERATION_MODE_MANUAL)
        gpu_devices = dcgm_agent.dcgmGetAllSupportedDevices(dcgm_handle)
        for gpu_device in gpu_devices:
            device_atrributes = dcgm_agent.dcgmGetDeviceAttributes(
                dcgm_handle, gpu_device).identifiers
            pci_bus_id = bytes(
                device_atrributes.pciBusId.decode('ascii').upper(),
                encoding='ascii')
            device_uuid = device_atrributes.uuid
            if pci_bus_id == bus_id:
                gpu_device = GPUDevice(gpu_device, bus_id, device_uuid)
                dcgm_agent.dcgmShutdown()
                return gpu_device
        else:
            dcgm_agent.dcgmShutdown()
            raise TritonModelAnalyzerException(
                f'GPU with {bus_id} bus id is not supported by DCGM.')

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
        pci_domain_id = device_identity['pci_domain_id']
        pci_device_id = device_identity['pci_device_id']
        pci_bus_id = device_identity['pci_bus_id']
        device_bus_id = \
            f'{pci_domain_id:08X}:{pci_bus_id:02X}:{pci_device_id:02X}.0'

        return GPUDeviceFactory.create_device_by_bus_id(
            bytes(device_bus_id, encoding='ascii'))

    @staticmethod
    def create_device_by_uuid(uuid, dcgmPath=None):
        """
        Create a GPU device using the GPU uuid.

        Parameters
        ----------
        uuid : str
            index of the device in the list of visible CUDA devices.

        Returns
        -------
        Device
            The device associated with the uuid.

        Raises
        ------
        TritonModelAnalyzerExcpetion
            If the uuid does not exist this exception will be raised.
        """

        structs._dcgmInit(dcgmPath)
        dcgm_agent.dcgmInit()

        # Start DCGM in the embedded mode to use the shared library
        dcgm_handle = dcgm_agent.dcgmStartEmbedded(
            structs.DCGM_OPERATION_MODE_MANUAL)
        gpu_devices = dcgm_agent.dcgmGetAllSupportedDevices(dcgm_handle)
        for gpu_device in gpu_devices:
            device_atrributes = dcgm_agent.dcgmGetDeviceAttributes(
                dcgm_handle, gpu_device).identifiers
            pci_bus_id = bytes(
                device_atrributes.pciBusId.decode('ascii').upper(),
                encoding='ascii')
            device_uuid = device_atrributes.uuid
            if bytes(uuid, encoding='ascii') == device_uuid:
                gpu_device = GPUDevice(gpu_device, pci_bus_id, device_uuid)
                dcgm_agent.dcgmShutdown()
                return gpu_device
        else:
            dcgm_agent.dcgmShutdown()
            raise TritonModelAnalyzerException(
                f'GPU UUID {uuid} was not found.')

    @staticmethod
    def verify_requested_gpus(requested_gpus):
        """
        Creates a list of GPU UUIDs corresponding to the GPUs visible to
        numba.cuda among the requested gpus

        Returns
        -------
        List
            list of uuids corresponding to visible GPUs among requested
        """

        cuda_visible_gpus = GPUDeviceFactory.get_cuda_visible_gpus().keys()

        if len(requested_gpus) == 1 and requested_gpus[0] == 'all':
            return list(cuda_visible_gpus)

        available_gpus = list(set(cuda_visible_gpus) & set(requested_gpus))
        return available_gpus

    @staticmethod
    def get_cuda_visible_gpus():
        """
        Gets the gpus visible to cuda

        Returns
        -------
        dict
            keys are gpu uuids
            values are device ids on this machine
        """

        cuda_visible_gpus = {}
        if numba.cuda.is_available():
            devices = numba.cuda.list_devices()
            for device in devices:
                gpu_device = GPUDeviceFactory.create_device_by_cuda_index(
                    device.id)
                cuda_visible_gpus[str(gpu_device.device_uuid(),
                                      encoding='ascii')] = str(
                                          gpu_device.device_id())
        return cuda_visible_gpus
