# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from .monitor import Monitor
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.record.types.gpu_utilization import GPUUtilization
from model_analyzer.record.types.gpu_used_memory import GPUUsedMemory
from model_analyzer.record.types.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.types.gpu_power_usage import GPUPowerUsage

from prometheus_client.parser import text_string_to_metric_families
import requests
import logging

logger = logging.getLogger(LOGGER_NAME)


class RemoteMonitor(Monitor):
    """
    Requests metrics from Triton's metrics
    endpoint
    """

    gpu_metrics = {
        'nv_gpu_utilization': GPUUtilization,
        'nv_gpu_memory_used_bytes': GPUUsedMemory,
        'nv_gpu_power_usage': GPUPowerUsage,
        'nv_gpu_memory_total_bytes': GPUFreeMemory
    }

    def __init__(self, metrics_url, frequency, metrics):
        super().__init__(frequency, metrics)
        self._metrics_url = metrics_url
        self._metrics_responses = []

        allowed_metrics = set(self.gpu_metrics.values())
        if not set(metrics).issubset(allowed_metrics):
            unsupported_metrics = set(metrics) - allowed_metrics
            raise TritonModelAnalyzerException(
                f"GPU monitoring does not currently support the following metrics: {unsupported_metrics}]"
            )

    def is_monitoring_connected(self) -> bool:
        try:
            status_code = requests.get(self._metrics_url, timeout=10).status_code
        except Exception as ex:
            return False

        return status_code == requests.codes["okay"]

    def _monitoring_iteration(self):
        """
        When this function runs, it requests all the metrics
        that triton has collected and organizes them into
        the dict. This function should run as fast
        as possible
        """

        self._metrics_responses.append(
            str(requests.get(self._metrics_url, timeout=10).content, encoding='ascii'))

    def _collect_records(self):
        """
        This function will organize the metrics responses
        and create Records out of them
        """

        records = []

        for response in self._metrics_responses:
            metrics = text_string_to_metric_families(response)
            processed_gpu_used_memory = False
            calculate_free_memory_after_pass = False
            gpu_memory_used_bytes = None
            for metric in metrics:
                if metric.name in self.gpu_metrics and self.gpu_metrics[
                        metric.name] in self._metrics:
                    for sample in metric.samples:
                        if sample.name == 'nv_gpu_memory_used_bytes':
                            processed_gpu_used_memory = True
                            gpu_memory_used_bytes = sample.value
                            self._create_and_add_record(
                                records, sample, gpu_memory_used_bytes // 1.0e6)
                        elif sample.name == 'nv_gpu_memory_total_bytes':
                            if processed_gpu_used_memory:
                                self._create_and_add_record(
                                    records, sample,
                                    (sample.value - gpu_memory_used_bytes) //
                                    1.0e6)
                            else:
                                total_memory_metric = metric
                                calculate_free_memory_after_pass = True
                        elif sample.name == 'nv_gpu_utilization':
                            self._create_and_add_record(records, sample,
                                                        sample.value * 100)
                        else:
                            self._create_and_add_record(records, sample,
                                                        sample.value)
            if calculate_free_memory_after_pass:
                for sample in total_memory_metric.samples:
                    self._create_and_add_record(
                        records, sample,
                        (sample.value - gpu_memory_used_bytes) // 1.0e6)

        return records

    def _create_and_add_record(self, records, sample, sample_value):
        """
        Adds a record to given dict
        """

        records.append(self.gpu_metrics[sample.name](
            value=sample_value, device_uuid=sample.labels['gpu_uuid']))
