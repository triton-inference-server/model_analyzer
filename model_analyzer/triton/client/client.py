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

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

logger = logging.getLogger(__name__)


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

    def load_model(self, model):
        """
        Request the inference server to load
        a particular model in explicit model
        control mode.

        Parameters
        ----------
        model : Model
            model to load from repository

        Raises
        ------
        TritonModelAnalyzerException
            If server throws Exception
        """

        try:
            self._client.load_model(model.name())
        except Exception as e:
            raise TritonModelAnalyzerException(
                f"Unable to load the model : {e}")
        logger.info(f'Model {model.name()} loaded.')

    def unload_model(self, model):
        """
        Request the inference server to unload
        a particular model in explicit model
        control mode.

        Parameters
        ----------
        model : Model
            model to unload from repository

        Raises
        ------
        TritonModelAnalyzerException
            If server throws Exception
        """

        try:
            self._client.unload_model(model.name())
        except Exception as e:
            raise TritonModelAnalyzerException(
                f"Unable to unload the model : {e}")
        logger.info(f'Model {model.name()} unloaded.')

    def wait_for_model_ready(self, model, num_retries):
        """
        Returns when model is ready.

        Parameters
        ----------
        model : str
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
        while retries > 0:
            try:
                if self._client.is_model_ready(model.name()):
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
            "Could not determine model readiness. "
            "Number of retries exceeded.")
