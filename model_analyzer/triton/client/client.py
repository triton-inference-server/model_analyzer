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

import time

from tritonclient.utils import InferenceServerException
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

WAIT_FOR_READY_NUM_RETRIES = 100


class TritonClient:
    """
    Defines the interface for the objects created by
    TritonClientFactory
    """

    def wait_for_server_ready(self, num_retries=WAIT_FOR_READY_NUM_RETRIES):
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
        while num_retries > 0:
            try:
                if self._client.is_server_ready():
                    return
                else:
                    time.sleep(0.1)
                    num_retries -= 1
            except Exception as e:
                time.sleep(0.1)
                num_retries -= 1
                pass
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
            If server throws InferenceServerException
        """
        try:
            self._client.load_model(model.name())
        except InferenceServerException as e:
            raise TritonModelAnalyzerException(
                f"Unable to load the model : {e}")

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
            If server throws InferenceServerException
        """
        try:
            self._client.unload_model(model.name())
        except InferenceServerException as e:
            raise TritonModelAnalyzerException(
                f"Unable to unload the model : {e}")

    def wait_for_model_ready(
            self,
            model,
            num_retries=WAIT_FOR_READY_NUM_RETRIES):
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
        while num_retries > 0:
            try:
                if self._client.is_model_ready(model.name()):
                    return
                else:
                    time.sleep(0.05)
                    num_retries -= 1
            except Exception as e:
                time.sleep(0.05)
                num_retries -= 1
                pass
        raise TritonModelAnalyzerException(
            "Could not determine model readiness. "
            "Number of retries exceeded.")
