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

import argparse
import yaml


def _get_sweep_configs(profile_models):

    sweep_configs = []
    model_config = {
        'run_config_search_disable': True,
        'perf_analyzer_cpu_util': 600,
        'profile_models': {
            model: {
                'model_config_parameters': {
                    'instance_group': [{
                        'count': [1, 2, 3, 4],
                        'kind': 'KIND_GPU'
                    }]
                }
            }
            for model in profile_models
        },
        'triton_launch_mode': ['docker'],
    }
    model_config['total_param'] = 4
    sweep_configs.append(model_config)

    model_config = {
        'run_config_search_disable': True,
        'perf_analyzer_cpu_util': 600,
        'profile_models': {
            model: {
                'model_config_parameters': {
                    'dynamic_batching': [{}, None],
                    'instance_group': [{
                        'count': [1],
                        'kind': ['KIND_GPU', None]
                    }]
                }
            }
            for model in profile_models
        },
        'triton_launch_mode': ['docker'],
    }
    model_config['total_param'] = 4
    sweep_configs.append(model_config)

    model_config = {
        'run_config_search_disable': True,
        'perf_analyzer_cpu_util': 600,
        'profile_models': {
            model: {
                'model_config_parameters': {
                    'dynamic_batching': {
                        'preferred_batch_size': [[4, 8], [5, 6]],
                        'max_queue_delay_microseconds': [100, 200]
                    }
                }
            }
            for model in profile_models
        },
        'triton_launch_mode': ['docker'],
    }
    model_config['total_param'] = 4
    sweep_configs.append(model_config)
    return sweep_configs


def get_all_configurations(profile_models):

    run_params = []
    run_params += _get_sweep_configs(profile_models)
    return run_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--profile-models',
                        type=str,
                        required=True,
                        help='The models to be profiled for this test')

    args = parser.parse_args()
    for i, configuration in enumerate(get_all_configurations(args.profile_models.split(','))):
        total_param = configuration['total_param']
        del configuration['total_param']
        with open(f'./config-{i}.yml', 'w') as file:
            yaml.dump(configuration, file)
        with open(f'./config-{i}.txt', 'w') as file:
            file.write(str(total_param))
