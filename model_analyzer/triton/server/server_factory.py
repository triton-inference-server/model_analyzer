# Copyright (c) 2020,21 NVIDIA CORPORATION. All rights reserved.
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

from model_analyzer.device.gpu_device_factory import GPUDeviceFactory

from .server_docker import TritonServerDocker
from .server_local import TritonServerLocal


class TritonServerFactory:
    """
    A factory for creating TritonServer instances
    """
    @staticmethod
    def create_server_docker(image, config, gpus, log_path=None, mounts=None):
        """
        Parameters
        ----------
        image : str
            The tritonserver docker image to pull and run
        config : TritonServerConfig
            the config object containing arguments for this server instance
        gpus : list of str
            List of GPU UUIDs to be mounted and used in the container
            Use ["all"] to include all GPUs
        log_path: str
            Absolute path to the triton log file
        mounts: list of str
            The volumes to be mounted to the tritonserver container

        Returns
        -------
        TritonServerDocker
        """

        return TritonServerDocker(
            image=image,
            config=config,
            gpus=GPUDeviceFactory.verify_requested_gpus(gpus),
            log_path=log_path,
            mounts=mounts)

    @staticmethod
    def create_server_local(path, config, gpus, log_path=None):
        """
        Parameters
        ----------
        path  : str
            The absolute path to the tritonserver executable
        config : TritonServerConfig
            the config object containing arguments for this server instance
        gpus: list of str
            List of GPU UUIDs to be made visible to Triton
            Use ["all"] to include all GPUs
        log_path: str
            Absolute path to the triton log file

        Returns
        -------
        TritonServerLocal
        """

        return TritonServerLocal(
            path=path,
            config=config,
            gpus=GPUDeviceFactory.verify_requested_gpus(gpus),
            log_path=log_path)
