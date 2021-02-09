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

from itertools import product
import yaml


def get_random_intervals():
    # Beginning of the interval
    begin = 10

    # End of the interval
    end = 12

    # Step size
    step = 2

    return list(range(begin, end + 1, step)), begin, end, step


def get_all_configurations():
    intervals, begin, end, step = get_random_intervals()
    intervals = [str(x) for x in intervals]

    shared_args = {
        'model_names': [['classification_breast_v1']],
        'triton_launch_mode': ['docker'],
        'batch_sizes': [{
            'start': begin,
            'stop': end,
            'step': step
        }],
        'concurrency': [{
            'start': begin,
            'stop': end,
            'step': step
        }],
    }

    # model names combinations
    param_combs = []
    model_names_combs = dict(shared_args)
    model_names_combs['model_names'] = [
        ['classification_breast_v1', 'classification_chestxray_v1'],
        ['classification_chestxray_v1'],
        'classification_breast_v1,classification_chestxray_v1'
    ]

    param_combs += list(product(*tuple(model_names_combs.values())))

    range_combs = [{
        'start': begin,
        'stop': end,
        'step': step
    }, intervals, ','.join(intervals), '1']

    # concurrency value combinations
    concurrency_combs = dict(shared_args)
    concurrency_combs['concurrency'] = range_combs
    param_combs += list(product(*tuple(concurrency_combs.values())))

    # batch size value combinations
    batch_size_combs = dict(shared_args)
    batch_size_combs['batch_sizes'] = range_combs
    param_combs += list(product(*tuple(batch_size_combs.values())))

    run_params = []
    for param_combination in param_combs:
        new_run = dict(zip(shared_args.keys(), param_combination))

        if type(new_run['model_names']) is str:
            number_of_models = len(new_run['model_names'].split(','))
        else:
            number_of_models = len(new_run['model_names'])

        if type(new_run['concurrency']) is str:
            concurrency_number = len(new_run['concurrency'].split(','))
        else:
            concurrency_number = len(intervals)

        if type(new_run['batch_sizes']) is str:
            batch_size_number = len(new_run['batch_sizes'].split(','))
        else:
            batch_size_number = len(intervals)
        new_run['total_param'] = \
            number_of_models * concurrency_number * batch_size_number
        run_params.append(new_run)

    return run_params


if __name__ == "__main__":
    for i, configuration in enumerate(get_all_configurations()):
        total_param = configuration['total_param']
        del configuration['total_param']
        with open(f'./config-{i}.yml', 'w') as file:
            yaml.dump(configuration, file)
        with open(f'./config-{i}.txt', 'w') as file:
            file.write(str(total_param))
