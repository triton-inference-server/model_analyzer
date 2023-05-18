# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from typing import Dict, Any, List, Optional
from copy import deepcopy

from numba import cuda
from distutils.dir_util import copy_tree
from google.protobuf import text_format, json_format
from google.protobuf.descriptor import FieldDescriptor
from tritonclient.grpc import model_config_pb2
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from model_analyzer.triton.server.server_factory import TritonServerFactory
from model_analyzer.config.input.objects.config_model_profile_spec import ConfigModelProfileSpec
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.device.gpu_device import GPUDevice


class ModelConfig:
    """
    A class that encapsulates all the metadata about a Triton model.
    """

    _default_config_dict: Dict[str, Any] = {}

    def __init__(self, model_config):
        """
        Parameters
        -------
        model_config : protobuf message
        """

        self._model_config = model_config
        self._cpu_only = False

    def to_dict(self):
        model_config_dict = json_format.MessageToDict(self._model_config)
        model_config_dict['cpu_only'] = self._cpu_only
        return model_config_dict

    @classmethod
    def from_dict(cls, model_config_dict):
        if 'cpu_only' in model_config_dict:
            cpu_only = model_config_dict['cpu_only']
            del model_config_dict['cpu_only']
            model_config = ModelConfig.create_from_dictionary(model_config_dict)
            model_config._cpu_only = cpu_only
        else:
            model_config = ModelConfig.create_from_dictionary(model_config_dict)

        return model_config

    @staticmethod
    def create_model_config_dict(config, client, gpus, model_repository,
                                 model_name):
        """ 
        Attempts to create a base model config dict from config.pbtxt, if one exists
        If the config.pbtxt is not present, we will load a Triton Server with the 
        base model and have it create a default config for MA, if possible

        Parameters:
        -----------
        config: ModelAnalyzerConfig
        client: TritonClient
        gpus: List of GPUDevices
        model_repository: str
            path to the model repository on the file system
        model_name: str
            name of the base model
        """

        if (ModelConfig._default_config_dict and
                model_name in ModelConfig._default_config_dict):
            return deepcopy(ModelConfig._default_config_dict[model_name])

        model_path = f'{model_repository}/{model_name}'

        try:
            config = ModelConfig._create_from_file(model_path).get_config()
        except:
            if (config.triton_launch_mode == 'docker' or
                    config.triton_launch_mode == 'local'):
                config = ModelConfig._get_default_config_from_server(
                    config, client, gpus, model_name, model_path)
            else:
                if not os.path.exists(model_path):
                    raise TritonModelAnalyzerException(
                        f'Model path "{model_path}" specified does not exist.')

                if os.path.isfile(model_path):
                    raise TritonModelAnalyzerException(
                        f'Model output path "{model_path}" must be a directory.'
                    )

                model_config_path = os.path.join(model_path, "config.pbtxt")
                raise TritonModelAnalyzerException(
                    f'Path "{model_config_path}" does not exist.'
                    f' Triton does not support default config creation for {config.triton_launch_mode} mode.'
                )

        ModelConfig._default_config_dict[model_name] = config
        return deepcopy(config)

    @staticmethod
    def _get_default_config_from_server(config, client, gpus, model_name,
                                        model_path):
        """ 
        Load a Triton Server with the base model and have it create 
        a default config for MA, if possible

        Parameters:
        -----------
        config: ModelAnalyzerConfig
        client: TritonClient
        gpus: List of GPUDevices
        model_name: str
            name of the base model
        model_path : str
            path to the base model
        """

        server = TritonServerFactory.get_server_handle(
            config, gpus, use_model_repository=True)

        server.start()
        client.wait_for_server_ready(num_retries=config.client_max_retries,
                                     log_file=server.log_file())

        if (client.load_model(model_name) == -1):
            server.stop()

            if not os.path.exists(model_path):
                raise TritonModelAnalyzerException(
                    f'Model path "{model_path}" specified does not exist.')

            if os.path.isfile(model_path):
                raise TritonModelAnalyzerException(
                    f'Model output path "{model_path}" must be a directory.')

            model_config_path = os.path.join(model_path, "config.pbtxt")
            raise TritonModelAnalyzerException(
                f'Path "{model_config_path}" does not exist.'
                ' Attempted have Triton create a default config, but this is not'
                ' possible for this model type.')

        client.wait_for_model_ready(model_name, config.client_max_retries)

        config = client.get_model_config(model_name, config.client_max_retries)

        server.stop()

        if ("input" not in config or "output" not in config):
            model_config_path = os.path.join(model_path, "config.pbtxt")
            raise TritonModelAnalyzerException(
                f'Path "{model_config_path}" does not exist.'
                ' Attempted have Triton create a default config, but this is not'
                ' possible for this model type.')

        return config

    @staticmethod
    def _create_from_file(model_path):
        """
        Constructs a ModelConfig from the pbtxt at file

        Parameters
        -------
        model_path : str
            The full path to this model directory

        Returns
        -------
        ModelConfig
        """

        if not os.path.exists(model_path):
            raise TritonModelAnalyzerException(
                f'Model path "{model_path}" specified does not exist.')

        if os.path.isfile(model_path):
            raise TritonModelAnalyzerException(
                f'Model output path "{model_path}" must be a directory.')

        model_config_path = os.path.join(model_path, "config.pbtxt")
        if not os.path.isfile(model_config_path):
            raise TritonModelAnalyzerException(
                f'Path "{model_config_path}" does not exist.'
                ' Make sure that you have specified the correct model'
                ' repository and model name(s).')

        with open(model_config_path, 'r+') as f:
            config_str = f.read()

        protobuf_message = text_format.Parse(config_str,
                                             model_config_pb2.ModelConfig())

        return ModelConfig(protobuf_message)

    @staticmethod
    def create_from_dictionary(model_dict):
        """
        Constructs a ModelConfig from a Python dictionary

        Parameters
        -------
        model_dict : dict
            A dictionary containing the model configuration.

        Returns
        -------
        ModelConfig
        """

        protobuf_message = json_format.ParseDict(model_dict,
                                                 model_config_pb2.ModelConfig())

        return ModelConfig(protobuf_message)

    @staticmethod
    def create_from_triton_api(client, model_name, num_retries):
        """
        Creates the model config from the Triton API.

        Parameters
        ----------
        client : TritonClient
            Triton client to use to call the API
        model_name : str
            Name of the model to request config for.
        num_retries : int
            Number of times to try loading the model.
        """

        model_config_dict = client.get_model_config(model_name, num_retries)

        return ModelConfig.create_from_dictionary(model_config_dict)

    @staticmethod
    def create_from_profile_spec(spec: ConfigModelProfileSpec,
                                 config: ConfigCommandProfile,
                                 client: TritonClient,
                                 gpus: List[GPUDevice]) -> "ModelConfig":
        """
        Creates the model config from a ModelProfileSpec, plus assoc. collateral 
        """

        model_config_dict = ModelConfig.create_model_config_dict(
            config=config,
            client=client,
            gpus=gpus,
            model_repository=config.model_repository,
            model_name=spec.model_name())

        model_config = ModelConfig.create_from_dictionary(model_config_dict)

        return model_config

    def set_cpu_only(self, cpu_only):
        """
        Parameters
        ----------
        bool
            Whether this model config has only
            CPU instance groups
        """

        self._cpu_only = cpu_only

    def cpu_only(self):
        """
        Returns
        -------
        bool
            Whether the model should be run on CPU only
        """

        return self._cpu_only

    def is_ensemble(self) -> bool:
        """
        Returns
        -------
        bool
           True if this is an ensemble model
        """

        return getattr(self._model_config, "platform") == "ensemble"

    def get_ensemble_composing_models(self) -> Optional[List[str]]:
        """
        Returns
        -------
            List[str]: Sub-model names
        """

        if not self.is_ensemble():
            raise TritonModelAnalyzerException(
                "Cannot find composing_models. Model platform is not ensemble.")

        try:
            composing_models = [
                model['modelName']
                for model in self.to_dict()['ensembleScheduling']['step']
            ]
        except:
            raise TritonModelAnalyzerException(
                "Cannot find composing_models. Ensemble Scheduling and/or step is not present in config protobuf."
            )

        return composing_models

    def set_composing_model_variant_name(self, composing_model_name: str,
                                         variant_name: str) -> None:
        """
        Replaces the Ensembles composing_model's name with the variant name
        """

        if not self.is_ensemble():
            raise TritonModelAnalyzerException(
                "Cannot find composing_models. Model platform is not ensemble.")

        model_config_dict = self.to_dict()

        try:
            for composing_model in model_config_dict['ensembleScheduling'][
                    'step']:
                if composing_model['modelName'] == composing_model_name:
                    composing_model['modelName'] = variant_name
        except:
            raise TritonModelAnalyzerException(
                "Cannot find composing_models. Ensemble Scheduling and/or step is not present in config protobuf."
            )

        self._model_config = self.from_dict(model_config_dict)._model_config

    def set_model_name(self, model_name: str) -> None:
        model_config_dict = self.to_dict()
        model_config_dict['name'] = model_name
        self._model_config = self.from_dict(model_config_dict)._model_config

    def write_config_to_file(self, model_path, src_model_path,
                             first_variant_model_path):
        """
        Writes a protobuf config file.

        Parameters
        ----------
        model_path : str
            Path to write the model config.

        src_model_path : str
            Path to the source model in the Triton Model Repository

        first_variant_model_path : str
            Indicates the path to the first model variant.

        Raises
        ------
        TritonModelAnalyzerException
            If the path doesn't exist or the path is a file
        """

        if not os.path.exists(model_path):
            raise TritonModelAnalyzerException(
                'Output path specified does not exist.')

        if os.path.isfile(model_path):
            raise TritonModelAnalyzerException(
                'Model output path must be a directory.')

        model_config_bytes = text_format.MessageToBytes(self._model_config)
        # Create current variant model as symlinks to first variant model
        if first_variant_model_path is not None:
            for file in os.listdir(first_variant_model_path):
                # Do not copy the config.pbtxt file
                if file == 'config.pbtxt':
                    continue
                else:
                    os.symlink(
                        os.path.join(
                            os.path.relpath(first_variant_model_path,
                                            model_path), file),
                        os.path.join(model_path, file))
        else:
            # Create first variant model as copy of source model
            copy_tree(src_model_path, model_path)

        with open(os.path.join(model_path, "config.pbtxt"), 'wb') as f:
            f.write(model_config_bytes)

    def get_config(self):
        """
        Get the model config.

        Returns
        -------
        dict
            A dictionary containing the model configuration.
        """

        return json_format.MessageToDict(self._model_config,
                                         preserving_proto_field_name=True)

    def set_config(self, config):
        """
        Set the model config from a dictionary.

        Parameters
        ----------
        config : dict
            The new dictionary containing the model config.
        """

        self._model_config = json_format.ParseDict(
            config, model_config_pb2.ModelConfig())

    def set_field(self, name, value):
        """
        Set a value for a Model Config field.

        Parameters
        ----------
        name : str
            Name of the field
        value : object
            The value to be used for the field.
        """
        model_config = self._model_config

        if model_config.DESCRIPTOR.fields_by_name[
                name].label == FieldDescriptor.LABEL_REPEATED:
            repeated_field = getattr(model_config, name)
            del repeated_field[:]
            repeated_field.extend(value)
        else:
            setattr(model_config, name, value)

    def get_field(self, name):
        """
        Get the value for the current field.
        """

        model_config = self._model_config
        return getattr(model_config, name)

    def max_batch_size(self) -> int:
        """
        Returns the max batch size (int)
        """

        model_config = self.get_config()
        return model_config.get('max_batch_size', 0)

    def dynamic_batching_string(self) -> str:
        """
        Returns
        -------
        str
            representation of the dynamic batcher
            configuration used to generate this result
        """

        model_config = self.get_config()
        if 'dynamic_batching' in model_config:
            return "Enabled"
        else:
            return "Disabled"

    def instance_group_string(self, system_gpu_count: int) -> str:
        """
        Returns
        -------
        str
            representation of the instance group used 
            to generate this result
        """

        model_config = self.get_config()

        # TODO change when remote mode is fixed
        default_kind = 'GPU' if cuda.is_available() else 'CPU'
        default_count = 1

        instance_group_list: List[Dict[str, Any]] = [{}]
        if 'instance_group' in model_config:
            instance_group_list = model_config['instance_group']

        kind_to_count: Dict[str, Any] = {}

        for group in instance_group_list:
            group_kind = default_kind
            group_count = default_count
            group_gpus_count = system_gpu_count
            # Update with instance group values
            if 'kind' in group:
                group_kind = group['kind'].split('_')[1]
            if 'count' in group:
                group_count = group['count']
            if 'gpus' in group:
                group_gpus_count = len(group['gpus'])

            group_total_count = group_count
            if group_kind == "GPU":
                group_total_count *= group_gpus_count

            if group_kind not in kind_to_count:
                kind_to_count[group_kind] = 0
            kind_to_count[group_kind] += group_total_count

        ret_str = ""
        for k, v in kind_to_count.items():
            if ret_str != "":
                ret_str += " + "
            ret_str += f'{v}:{k}'
        return ret_str
