# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import yaml


def _get_sweep_configs():

    sweep_configs = []
    model_config = {
        'analysis_models': ['vgg19_libtorch'],
        'server_output_fields':
        ['model_name', 'gpu_id', 'gpu_used_memory', 'gpu_utilization'],
        'gpu_output_fields': [
            'model_name', 'satisfies_constraints', 'gpu_used_memory',
            'gpu_utilization', 'gpu_power_usage'
        ],
        'inference_output_fields': [
            'model_name', 'batch_size', 'concurrency', 'model_config_path',
            'perf_throughput', 'perf_latency', 'cpu_used_ram'
        ]
    }

    model_config['total_param_server'] = len(
        model_config['server_output_fields'])
    model_config['total_param_gpu'] = len(model_config['gpu_output_fields'])
    model_config['total_param_inference'] = len(
        model_config['inference_output_fields'])
    sweep_configs.append(model_config)
    return sweep_configs


def get_all_configurations():

    run_params = []
    run_params += _get_sweep_configs()
    return run_params


if __name__ == "__main__":
    for i, configuration in enumerate(get_all_configurations()):
        total_param_server = configuration['total_param_server']
        total_param_gpu = configuration['total_param_gpu']
        total_param_inference = configuration['total_param_inference']
        del configuration['total_param_server']
        del configuration['total_param_gpu']
        del configuration['total_param_inference']
        with open(f'./config-{i}-param-server.txt', 'w') as file:
            file.write(str(total_param_server))
        with open(f'./config-{i}-param-gpu.txt', 'w') as file:
            file.write(str(total_param_gpu))
        with open(f'./config-{i}-param-inference.txt', 'w') as file:
            file.write(str(total_param_inference))
        with open(f'./config-{i}.yml', 'w') as file:
            yaml.dump(configuration, file)
