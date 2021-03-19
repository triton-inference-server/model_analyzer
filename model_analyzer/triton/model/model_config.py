# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from numba import cuda
from distutils.dir_util import copy_tree
from google.protobuf import text_format, json_format
from google.protobuf.descriptor import FieldDescriptor
from tritonclient.grpc import model_config_pb2
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class ModelConfig:
    """
    A class that encapsulates all the metadata about a Triton model.
    """
    def __init__(self, model_config):
        """
        Parameters
        -------
        model_config : protobuf message
        """

        self._model_config = model_config

    @staticmethod
    def create_from_file(model_path):
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

        protobuf_message = json_format.ParseDict(
            model_dict, model_config_pb2.ModelConfig())

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

    def write_config_to_file(self,
                             model_path,
                             copy_original_model=False,
                             src_model_path=None):
        """
        Writes a protobuf config file.

        Parameters
        ----------
        model_path : str
            Path to write the model config.

        copy_original_model : bool
            Whether or not to copy the original model.

        Raises
        ------
        TritonModelAnalzyerException
            If the path doesn't exist or the path is a file
        """

        if not os.path.exists(model_path):
            raise TritonModelAnalyzerException(
                'Output path specified does not exist.')

        if os.path.isfile(model_path):
            raise TritonModelAnalyzerException(
                'Model output path must be a directory.')

        model_config_bytes = text_format.MessageToBytes(self._model_config)

        if copy_original_model:
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

    def dynamic_batching_string(self):
        """
        Returns
        -------
        str
            representation of the dynamic batcher
            configuration used to generate this result
        """

        model_config = self.get_config()
        if 'dynamic_batching' in model_config:
            if 'preferred_batch_size' in model_config['dynamic_batching']:
                dynamic_batch_sizes = model_config['dynamic_batching'][
                    'preferred_batch_size']
            else:
                dynamic_batch_sizes = [model_config['max_batch_size']]
            return f"[{' '.join([str(x) for x in dynamic_batch_sizes])}]"
        else:
            return "Disabled"

    def instance_group_string(self):
        """
        Returns
        -------
        str
            representation of the instance group used 
            to generate this result
        """

        model_config = self.get_config()

        # TODO change when remote mode is fixed
        # Set default count/kind
        count = 1
        if cuda.is_available():
            kind = 'GPU'
        else:
            kind = 'CPU'

        if 'instance_group' in model_config:
            instance_group_list = model_config['instance_group']
            group_str_list = []
            for group in instance_group_list:
                group_kind, group_count = kind, count
                # Update with instance group values
                if 'kind' in group:
                    group_kind = group['kind'].split('_')[1]
                if 'count' in group:
                    group_count = group['count']
                group_str_list.append(f"{group_count}/{group_kind}")
            return ','.join(group_str_list)
        return f"{count}/{kind}"
