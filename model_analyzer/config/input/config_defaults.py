# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# Common defaults
#

DEFAULT_CHECKPOINT_DIRECTORY = os.path.join(os.getcwd(), 'checkpoints')
DEFAULT_ONLINE_OBJECTIVES = {'perf_throughput': 10}
DEFAULT_OFFLINE_OBJECTIVES = {'perf_throughput': 10}
DEFAULT_MODEL_WEIGHTING = 1

#
# Profile Config defaults
#

DEFAULT_MONITORING_INTERVAL = 1.0
DEFAULT_DURATION_SECONDS = 3
DEFAULT_COLLECT_CPU_METRICS = False
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_GPUS = 'all'
DEFAULT_SKIP_SUMMARY_REPORTS = False
DEFAULT_SKIP_DETAILED_REPORTS = False
DEFAULT_OUTPUT_MODEL_REPOSITORY = os.path.join(os.getcwd(),
                                               'output_model_repository')
DEFAULT_OVERRIDE_OUTPUT_REPOSITORY_FLAG = False
DEFAULT_BATCH_SIZES = 1
DEFAULT_MAX_RETRIES = 50
DEFAULT_CLIENT_PROTOCOL = 'grpc'
DEFAULT_RUN_CONFIG_MAX_CONCURRENCY = 1024
DEFAULT_RUN_CONFIG_MIN_CONCURRENCY = 1
DEFAULT_RUN_CONFIG_MAX_REQUEST_RATE = 8192
DEFAULT_RUN_CONFIG_MIN_REQUEST_RATE = 16
DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT = 5
DEFAULT_RUN_CONFIG_MIN_INSTANCE_COUNT = 1
DEFAULT_RUN_CONFIG_MIN_MODEL_BATCH_SIZE = 1
DEFAULT_RUN_CONFIG_MAX_MODEL_BATCH_SIZE = 128
DEFAULT_RUN_CONFIG_MAX_BINARY_SEARCH_STEPS = 5
DEFAULT_RUN_CONFIG_SEARCH_DISABLE = False
DEFAULT_RUN_CONFIG_SEARCH_MODE = 'brute'
DEFAULT_RUN_CONFIG_PROFILE_MODELS_CONCURRENTLY_ENABLE = False
DEFAULT_REQUEST_RATE_SEARCH_ENABLE = False
DEFAULT_TRITON_LAUNCH_MODE = 'local'
DEFAULT_TRITON_DOCKER_IMAGE = 'nvcr.io/nvidia/tritonserver:23.07-py3'
DEFAULT_TRITON_HTTP_ENDPOINT = 'localhost:8000'
DEFAULT_TRITON_GRPC_ENDPOINT = 'localhost:8001'
DEFAULT_TRITON_METRICS_URL = 'http://localhost:8002/metrics'
DEFAULT_TRITON_SERVER_PATH = 'tritonserver'
DEFAULT_TRITON_INSTALL_PATH = '/opt/tritonserver'
DEFAULT_PERF_ANALYZER_TIMEOUT = 600
DEFAULT_PERF_ANALYZER_CPU_UTIL = 80.0
DEFAULT_PERF_ANALYZER_PATH = 'perf_analyzer'
DEFAULT_PERF_OUTPUT_FLAG = False
DEFAULT_PERF_MAX_AUTO_ADJUSTS = 10
DEFAULT_MEASUREMENT_MODE = 'count_windows'

DEFAULT_ONLINE_PLOTS = {
    'throughput_v_latency': {
        'title': 'Throughput vs. Latency',
        'x_axis': 'perf_latency_p99',
        'y_axis': 'perf_throughput',
        'monotonic': True
    },
    'gpu_mem_v_latency': {
        'title': 'GPU Memory vs. Latency',
        'x_axis': 'perf_latency_p99',
        'y_axis': 'gpu_used_memory',
        'monotonic': False
    }
}

DEFAULT_OFFLINE_PLOTS = {
    'through_v_batch_size': {
        'title': 'Throughput vs. Batch Size',
        'x_axis': 'batch_size',
        'y_axis': 'perf_throughput',
        'monotonic': False
    }
}

DEFAULT_CPU_MEM_PLOT = {
    'cpu_mem_v_latency': {
        'title': 'CPU Memory vs. Latency',
        'x_axis': 'perf_latency_p99',
        'y_axis': 'cpu_used_ram',
        'monotonic': False
    }
}

DEFAULT_EXPORT_PATH = os.getcwd()
DEFAULT_FILENAME_MODEL_INFERENCE = 'metrics-model-inference.csv'
DEFAULT_FILENAME_MODEL_GPU = 'metrics-model-gpu.csv'
DEFAULT_FILENAME_SERVER_ONLY = 'metrics-server-only.csv'

DEFAULT_INFERENCE_OUTPUT_FIELDS = [
    'model_name', 'batch_size', 'concurrency', 'model_config_path',
    'instance_group', 'max_batch_size', 'satisfies_constraints',
    'perf_throughput', 'perf_latency_p99'
]
DEFAULT_REQUEST_RATE_INFERENCE_OUTPUT_FIELDS = [
    'model_name', 'batch_size', 'request_rate', 'model_config_path',
    'instance_group', 'max_batch_size', 'satisfies_constraints',
    'perf_throughput', 'perf_latency_p99'
]
DEFAULT_GPU_OUTPUT_FIELDS = [
    'model_name', 'gpu_uuid', 'batch_size', 'concurrency', 'model_config_path',
    'instance_group', 'satisfies_constraints', 'gpu_used_memory',
    'gpu_utilization', 'gpu_power_usage'
]
DEFAULT_REQUEST_RATE_GPU_OUTPUT_FIELDS = [
    'model_name', 'gpu_uuid', 'batch_size', 'request_rate', 'model_config_path',
    'instance_group', 'satisfies_constraints', 'gpu_used_memory',
    'gpu_utilization', 'gpu_power_usage'
]
DEFAULT_SERVER_OUTPUT_FIELDS = [
    'model_name', 'gpu_uuid', 'gpu_used_memory', 'gpu_utilization',
    'gpu_power_usage'
]

DEFAULT_NUM_CONFIGS_PER_MODEL = 3
DEFAULT_NUM_TOP_MODEL_CONFIGS = 0

#
# Report Config defaults
#

DEFAULT_REPORT_FORMAT = 'pdf'

DEFAULT_ONLINE_REPORT_PLOTS = {
    'gpu_mem_v_latency': {
        'title': 'GPU Memory vs. Latency',
        'x_axis': 'perf_latency_p99',
        'y_axis': 'gpu_used_memory',
        'monotonic': False
    },
    'gpu_util_v_latency': {
        'title': 'GPU Utilization vs. Latency',
        'x_axis': 'perf_latency_p99',
        'y_axis': 'gpu_utilization',
        'monotonic': False
    },
    'cpu_mem_v_latency': {
        'title': 'RAM Usage vs. Latency',
        'x_axis': 'perf_latency_p99',
        'y_axis': 'cpu_used_ram',
        'monotonic': False
    },
    'gpu_power_v_latency': {
        'title': 'GPU Power vs. Latency',
        'x_axis': 'perf_latency_p99',
        'y_axis': 'gpu_power_usage',
        'monotonic': False
    }
}

DEFAULT_OFFLINE_REPORT_PLOTS = {
    'throughput_v_batch_size': {
        'title': 'Throughput vs. Batch Size',
        'x_axis': 'batch_size',
        'y_axis': 'perf_throughput',
        'monotonic': False
    },
    'latency_v_batch_size': {
        'title': 'p99 Latency vs. Batch Size',
        'x_axis': 'batch_size',
        'y_axis': 'perf_latency_p99',
        'monotonic': False
    }
}
