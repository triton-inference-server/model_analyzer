#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import numba.cuda


class Monitor:
    """Monitor class is a parent class used for monitoring
    GPU devices.
    """

    def __init__(self, frequency):
        self._frequency = frequency
        self._gpus = numba.cuda.gpus

    def start_recording_metrics(self, tags):
        raise NotImplementedError('start_recording_metric Not Implemented')

    def stop_recording_metrics(self, tags):
        raise NotImplementedError('stop_recording_metric Not Implemented')
