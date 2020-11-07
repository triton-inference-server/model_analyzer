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

from unittest.mock import patch, Mock, MagicMock

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException


class MockInferenceServerException(Exception):
    """
    A mock object that can be used in
    place of an InferenceServerException
    """
    pass


class MockTritonClientMethods:
    """
    Mocks the tritonclient module functions 
    used in model_analyzer/triton/client
    Provides functions to check operation.
    """

    def __init__(self):
        client_attrs = {
            'load_model': MagicMock(),
            'unload_model': MagicMock(),
            'is_model_ready': MagicMock(return_value=True)
        }
        mock_http_client = Mock(**client_attrs)
        mock_grpc_client = Mock(**client_attrs)
        self.patcher_http_client = patch(
            'model_analyzer.triton.client.http_client.httpclient.InferenceServerClient',
            Mock(return_value=mock_http_client))
        self.patcher_grpc_client = patch(
            'model_analyzer.triton.client.grpc_client.grpcclient.InferenceServerClient',
            Mock(return_value=mock_grpc_client))
        self.patcher_inference_server_exception = patch(
            'model_analyzer.triton.client.client.InferenceServerException',
            MockInferenceServerException)
        self.http_mock = self.patcher_http_client.start()
        self.grpc_mock = self.patcher_grpc_client.start()
        self.exception_mock = self.patcher_inference_server_exception.start()

    def stop(self):
        """
        Destroy the mocked classes and
        functions
        """

        self.patcher_http_client.stop()
        self.patcher_grpc_client.stop()
        self.patcher_inference_server_exception.stop()

    def raise_exception_on_load(self):
        """
        Set load_model to throw
        InferenceServerException
        """

        self.grpc_mock.return_value.load_model.side_effect = self.exception_mock
        self.http_mock.return_value.load_model.side_effect = self.exception_mock

    def raise_exception_on_unload(self):
        """
        Set load_model to throw
        InferenceServerException
        """

        self.grpc_mock.return_value.unload_model.side_effect = self.exception_mock
        self.http_mock.return_value.unload_model.side_effect = self.exception_mock

    def reset(self):
        """
        Reset the client mocks
        """
        self.grpc_mock.return_value.load_model.side_effect = None
        self.http_mock.return_value.load_model.side_effect = None
        self.grpc_mock.return_value.unload_model.side_effect = None
        self.http_mock.return_value.unload_model.side_effect = None
