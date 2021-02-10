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
from google.protobuf import text_format, json_format
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
        model_config : dict
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
                'Model path specified does not exist.'
            )

        if os.path.isfile(model_path):
            raise TritonModelAnalyzerException(
                'Model output path must be a directory.')

        with open(os.path.join(model_path, "config.pbtxt"), 'r+') as f:
            config_str = f.read()

        print(config_str)
        protobuf_message = text_format.Parse(
            config_str, model_config_pb2.ModelConfig())

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

    def write_config_to_file(self, model_path):
        """
        Writes a protobuf config file.

        Parameters
        ----------
        model_path : str
            Path to write the model config.

        Raises
        ------
        TritonModelAnalzyerException
            If the path doesn't exist or the path is a file
        """

        if not os.path.exists(model_path):
            raise TritonModelAnalyzerException(
                'Output path specified does not exist.'
            )

        if os.path.isfile(model_path):
            raise TritonModelAnalyzerException(
                'Model output path must be a directory.')

        model_config_bytes = text_format.MessageToBytes(
            self._model_config)

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
