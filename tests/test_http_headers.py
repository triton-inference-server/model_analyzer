#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

import unittest

from model_analyzer.triton.client.client_factory import TritonClientFactory

from .common import test_result_collector as trc
from .mocks.mock_client import MockTritonClientMethods


class TestHTTPHeaders(trc.TestResultCollector):
    """
    Test that HTTP headers are properly passed through the client stack
    """

    def setUp(self):
        self.tritonclient_mock = MockTritonClientMethods()
        self.tritonclient_mock.start()

    def tearDown(self):
        self.tritonclient_mock.stop()

    def test_http_client_with_headers(self):
        test_headers = {
            "Authorization": "Bearer test_token",
            "X-Custom-Header": "custom_value",
        }

        client = TritonClientFactory.create_http_client(
            server_url="http://localhost:8000", headers=test_headers
        )

        self.assertEqual(client._headers, test_headers)

    def test_http_client_without_headers(self):
        client = TritonClientFactory.create_http_client(
            server_url="http://localhost:8000"
        )

        self.assertEqual(client._headers, {})

    def test_grpc_client_has_empty_headers(self):
        client = TritonClientFactory.create_grpc_client(server_url="localhost:8001")

        self.assertEqual(client._headers, {})

    def test_load_model_with_headers(self):
        test_headers = {"X-API-Key": "secret123"}

        client = TritonClientFactory.create_http_client(
            server_url="http://localhost:8000", headers=test_headers
        )

        client.load_model("test_model")

        self.tritonclient_mock.http_mock.return_value.load_model.assert_called_with(
            "test_model", config=None, headers=test_headers
        )

    def test_unload_model_with_headers(self):
        """Test that unload_model passes headers correctly"""
        test_headers = {"X-API-Key": "secret456"}

        client = TritonClientFactory.create_http_client(
            server_url="http://localhost:8000", headers=test_headers
        )

        client.unload_model("test_model")

        self.tritonclient_mock.http_mock.return_value.unload_model.assert_called_with(
            "test_model", headers=test_headers
        )

    def test_is_server_ready_with_headers(self):
        test_headers = {"Authorization": "Bearer xyz"}

        client = TritonClientFactory.create_http_client(
            server_url="http://localhost:8000", headers=test_headers
        )

        client.is_server_ready()

        self.tritonclient_mock.http_mock.return_value.is_server_ready.assert_called_with(
            headers=test_headers
        )


if __name__ == "__main__":
    unittest.main()
