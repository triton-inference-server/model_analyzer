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

import unittest
import time
import sys
sys.path.append('../model_analyzer')


class TestNVMLMonitor(unittest.TestCase):

    def test_record_memory(self):
        from model_analyzer.monitor.nvml import NVMLMonitor

        # One measurement every 0.01 seconds
        frequency = 0.01
        monitoring_time = 10
        nvml_monitor = NVMLMonitor(frequency)
        nvml_monitor.start_recording_metrics(['memory'])
        time.sleep(monitoring_time)
        metrics = nvml_monitor.stop_recording_metrics()
        nvml_monitor.destroy()

        # Check that the interval monitored is almost 10 seconds
        self.assertAlmostEqual(metrics.get(-1).time - metrics.get(0).time,
                               monitoring_time, 1)

        # Check that the correct number of samples have been collected
        self.assertTrue((metrics.size() - monitoring_time / frequency) < 20)


if __name__ == '__main__':
    unittest.main()
