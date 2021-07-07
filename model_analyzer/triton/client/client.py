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

import time
import logging

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException


class TritonClient:
    """
    Defines the interface for the objects created by
    TritonClientFactory
    """
    def wait_for_server_ready(self, num_retries):
        """
        Parameters
        ----------
        num_retries : int
            number of times to send a ready status
            request to the server before raising
            an exception
        Raises
        ------
        TritonModelAnalyzerException
            If server readiness could not be
            determined in given num_retries
        """

        retries = num_retries
        while retries > 0:
            try:
                if self._client.is_server_ready():
                    return
                else:
                    time.sleep(0.05)
                    retries -= 1
            except Exception as e:
                time.sleep(0.05)
                retries -= 1
                if retries == 0:
                    raise TritonModelAnalyzerException(e)
        raise TritonModelAnalyzerException(
            "Could not determine server readiness. "
            "Number of retries exceeded.")

    def load_model(self, model_name):
        """
        Request the inference server to load
        a particular model in explicit model
        control mode.

        Parameters
        ----------
        model_name : str
            name of the model to load from repository

        Returns
        ------
        int or None
            Returns -1 if the failed.
        """

        try:
            self._client.load_model(model_name)
            logging.info(f'Model {model_name} loaded.')
        except Exception as e:
            logging.info(f'Model {model_name} load failed: {e}')
            return -1

    def unload_model(self, model_name):
        """
        Request the inference server to unload
        a particular model in explicit model
        control mode.

        Parameters
        ----------
        model_name : str
            name of the model to load from repository

        Raises
        ------
        TritonModelAnalyzerException
            If server throws Exception
        """

        try:
            self._client.unload_model(model_name)
            logging.info(f'Model {model_name} unloaded.')
        except Exception as e:
            logging.info(f'Model {model_name} unload failed: {e}')
            return -1

    def wait_for_model_ready(self, model_name, num_retries):
        """
        Returns when model is ready.

        Parameters
        ----------
        model_name : str
            name of the model to load from repository
        num_retries : int
            number of times to send a ready status
            request to the server before raising
            an exception

        Raises
        ------
        TritonModelAnalyzerException
            If could not determine model readiness
            in given num_retries
        """

        retries = num_retries
        error = None
        while retries > 0:
            try:
                if self._client.is_model_ready(model_name):
                    return
                else:
                    time.sleep(0.05)
                    retries -= 1
            except Exception as e:
                error = e
                time.sleep(0.05)
                retries -= 1

        logging.info(
            f'Model readiness failed for model {model_name}. Error {error}')
        return -1

    def get_model_config(self, model_name, num_retries):
        """
        Model name to get the config for.

        Parameters
        ----------
        model_name : str
            Name of the model to find the config.

        num_retries : int
            Number of times to wait for the model load

        Returns
        -------
        dict or None
            A dictionary containg the model config.
        """

        self.load_model(model_name)
        self.wait_for_model_ready(model_name, num_retries)
        model_config_dict = self._client.get_model_config(model_name)
        self.unload_model(model_name)
        return model_config_dict
