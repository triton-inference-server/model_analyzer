#!/usr/bin/env python3

# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import time
from subprocess import DEVNULL

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

logger = logging.getLogger(LOGGER_NAME)


class TritonClient:
    """
    Defines the interface for the objects created by
    TritonClientFactory
    """

    def wait_for_server_ready(
        self,
        num_retries,
        sleep_time=1,
        log_file=None,
    ):
        """
        Parameters
        ----------
        num_retries : int
            number of times to send a ready status
            request to the server before raising
            an exception
        sleep_time: int
            amount of time in seconds to sleep between retries
        log_file: TextIOWrapper
            file that contains the server's output log
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
                    time.sleep(sleep_time)
                    return
                else:
                    self._check_for_triton_log_errors(log_file)
                    time.sleep(sleep_time)
                    retries -= 1
            except Exception as e:
                self._check_for_triton_log_errors(log_file)
                time.sleep(sleep_time)
                retries -= 1
                if retries == 0:
                    raise TritonModelAnalyzerException(e)
        raise TritonModelAnalyzerException(
            "Could not determine server readiness. " "Number of retries exceeded."
        )

    def load_model(self, model_name, variant_name="", config_str=None):
        """
        Request the inference server to load
        a particular model in explicit model
        control mode.

        Parameters
        ----------
        model_name : str
            Name of the model

        variant_name: str
            Name of the model variant

        config_str: str
            Optional config string used to load the model

        Returns
        ------
        int or None
            Returns -1 if the failed.
        """

        variant_name = variant_name if variant_name else model_name

        try:
            self._client.load_model(model_name, config=config_str)
            logger.debug(f"Model {variant_name} loaded")
            return None
        except Exception as e:
            logger.info(f"Model {variant_name} load failed: {e}")
            if "polling is enabled" in e.message():
                raise TritonModelAnalyzerException(
                    "The remote Tritonserver needs to be launched in EXPLICIT mode"
                )
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

        Returns
        ------
        int or None
            Returns -1 if the failed.
        """

        try:
            self._client.unload_model(model_name)
            logger.debug(f"Model {model_name} unloaded")
            return None
        except Exception as e:
            logger.info(f"Model {model_name} unload failed: {e}")
            return -1

    def wait_for_model_ready(self, model_name, num_retries, sleep_time=1):
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

        Returns
        ------
        int or None
            Returns -1 if the failed.
        """

        retries = num_retries
        error = None
        while retries > 0:
            try:
                if self._client.is_model_ready(model_name):
                    return None
                else:
                    time.sleep(sleep_time)
                    retries -= 1
            except Exception as e:
                error = e
                time.sleep(sleep_time)
                retries -= 1

        logger.info(f"Model readiness failed for model {model_name}. Error {error}")
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
            A dictionary containing the model config.
        """

        self.wait_for_model_ready(model_name, num_retries)
        model_config_dict = self._client.get_model_config(model_name)
        return model_config_dict

    def is_server_ready(self):
        """
        Returns true if the server is ready. Else False
        """
        return self._client.is_server_ready()

    def _check_for_triton_log_errors(self, log_file):
        if not log_file or log_file == DEVNULL:
            return

        log_file.seek(0)
        log_output = log_file.read()

        if not type(log_output) == str:
            log_output = log_output.decode("utf-8")

        if log_output:
            if "Unexpected argument:" in log_output:
                error_start = log_output.find("Unexpected argument:")
                raise TritonModelAnalyzerException(
                    f"Error: TritonServer did not launch successfully\n\n{log_output[error_start:]}"
                )
