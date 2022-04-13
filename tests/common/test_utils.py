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

from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from model_analyzer.result.model_config_measurement import ModelConfigMeasurement
from model_analyzer.result.run_config_result import RunConfigResult
from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig

from model_analyzer.config.input.config_defaults import \
    DEFAULT_BATCH_SIZES, DEFAULT_TRITON_LAUNCH_MODE, DEFAULT_CLIENT_PROTOCOL, \
    DEFAULT_MEASUREMENT_MODE, DEFAULT_TRITON_GRPC_ENDPOINT, DEFAULT_TRITON_HTTP_ENDPOINT, \
    DEFAULT_TRITON_INSTALL_PATH, DEFAULT_OUTPUT_MODEL_REPOSITORY


def convert_to_bytes(string):
    """
    Converts string into bytes and ensures minimum length requirement 
    for compatibility with unpack function called in usr/lib/python3.8/gettext.py
    
    Parameters
    ----------
    string: str
    """
    if (len(string) > 4):
        return bytes(string, 'utf-8')
    else:
        return bytes(string + "    ", 'utf-8')


def convert_non_gpu_metrics_to_data(non_gpu_metric_values):
    """ 
    Non GPU data will be a dict whose keys and values are
    a list of Records
    
    Parameters
    ----------
    non_gpu_metric_values: dict of non-gpu metrics
    """

    non_gpu_data = []
    non_gpu_metric_tags = list(non_gpu_metric_values.keys())

    for i, metric in enumerate(
            MetricsManager.get_metric_types(non_gpu_metric_tags)):
        non_gpu_data.append(
            metric(value=non_gpu_metric_values[non_gpu_metric_tags[i]]))

    return non_gpu_data


def convert_gpu_metrics_to_data(gpu_metric_values):
    """
    GPU data will be a dict whose keys are gpu_ids and values
    are lists of Records
    
    Parameters
    ----------
    gpu_metric_values: dict of gpu metrics
    """
    gpu_data = {}
    for gpu_uuid, metrics_values in gpu_metric_values.items():
        gpu_data[gpu_uuid] = []
        gpu_metric_tags = list(metrics_values.keys())
        for i, gpu_metric in enumerate(
                MetricsManager.get_metric_types(gpu_metric_tags)):
            gpu_data[gpu_uuid].append(
                gpu_metric(value=metrics_values[gpu_metric_tags[i]]))

    return gpu_data


def convert_avg_gpu_metrics_to_data(avg_gpu_metric_values):
    """
    Avg GPU data will be a dict of Records
    
    Parameters
    ----------
    gpu_metric_values: dict of gpu metrics
    """
    avg_gpu_data = {}
    avg_gpu_metric_tags = list(avg_gpu_metric_values.keys())

    for i, avg_gpu_metric in enumerate(
            MetricsManager.get_metric_types(avg_gpu_metric_tags)):
        avg_gpu_data[avg_gpu_metric_tags[i]] = avg_gpu_metric(
            value=avg_gpu_metric_values[avg_gpu_metric_tags[i]])

    return avg_gpu_data


def construct_perf_analyzer_config(model_name='my-model',
                                   output_file_name='my-model',
                                   batch_size=DEFAULT_BATCH_SIZES,
                                   concurrency=1,
                                   launch_mode=DEFAULT_TRITON_LAUNCH_MODE,
                                   client_protocol=DEFAULT_CLIENT_PROTOCOL,
                                   perf_analyzer_flags=None):
    """
    Constructs a Perf Analyzer Config
    
    Parameters
    ----------
    model_name: str
        The name of the model
    output_file_name: str
        The name of the output file
    batch_size: int
        The batch size for this PA configuration
    concurrency: int
        The concurrency value for this PA configuration
    launch_mode: str
        The launch mode for this PA configuration
    client_protocol: str
        The client protocol for this PA configuration
    perf_analyzer_flags: dict
        A dict of any additional PA flags to be set
    
    Returns
    -------
    PerfAnalyzerConfig
        constructed with all of the above data.
    """

    pa_config = PerfAnalyzerConfig()
    pa_config._options['-m'] = model_name
    pa_config._options['-f'] = output_file_name
    pa_config._options['-b'] = batch_size
    pa_config._args['concurrency-range'] = concurrency
    pa_config._args['measurement-mode'] = DEFAULT_MEASUREMENT_MODE

    pa_config.update_config(perf_analyzer_flags)

    if launch_mode == 'c_api':
        pa_config._args['service-kind'] = 'triton_c_api'
        pa_config._args['triton-server-directory'] = DEFAULT_TRITON_INSTALL_PATH
        pa_config._args['model-repository'] = DEFAULT_OUTPUT_MODEL_REPOSITORY
    else:
        pa_config._options['-i'] = client_protocol
        if client_protocol == 'http':
            pa_config._options['-u'] = DEFAULT_TRITON_HTTP_ENDPOINT
        else:
            pa_config._options['-u'] = DEFAULT_TRITON_GRPC_ENDPOINT

    return pa_config


def construct_run_config_measurement(model_name, model_config_names,
                                     model_specific_pa_params,
                                     gpu_metric_values, non_gpu_metric_values,
                                     metric_objectives, model_config_weights):
    """
    Construct a RunConfig measurement from the given data

    Parameters
    ----------
    model_name: str
        The name of the model that generated this result
    model_config_names: list of str
        A list of Model Config names that generated this result
    model_specific_pa_params: list of dict
        A list (one per model config) of dict's of PA parameters that change
        between models in a multi-model run
    gpu_metric_values: dict
        Keys are gpu id, values are dict
        The dict where keys are gpu based metric tags, values are the data
    non_gpu_metric_values: list of dict
        List of (one per model config) dict's where keys are non gpu perf metrics, values are the data
    metric_objectives: list of RecordTypes
        A list of metric objectives (one per model config) used to compare measurements
    model_config_weights: list of ints
        A list of weights (one per model config) used to bias measurement results between models

    Returns
    -------
    RunConfigMeasurement
        constructed with all of the above data.
    """

    gpu_data = convert_gpu_metrics_to_data(gpu_metric_values)

    model_variants_name = ''.join(model_config_names)
    rc_measurement = RunConfigMeasurement(model_variants_name, gpu_data)

    non_gpu_data = [
        convert_non_gpu_metrics_to_data(non_gpu_metric_value)
        for non_gpu_metric_value in non_gpu_metric_values
    ]

    for index, model_config_name in enumerate(model_config_names):
        rc_measurement.add_model_config_measurement(
            model_config_name=model_config_name,
            model_specific_pa_params=model_specific_pa_params[index],
            non_gpu_data=non_gpu_data[index])

    rc_measurement.set_model_config_weighting(model_config_weights)
    rc_measurement.set_metric_weightings(metric_objectives)

    return rc_measurement


def construct_run_config_result(avg_gpu_metric_values,
                                avg_non_gpu_metric_values_list,
                                comparator,
                                value_step=1,
                                model_name="fake_model_name",
                                run_config=None):
    """
    Takes a dictionary whose values are average
    metric values, constructs artificial data 
    around these averages, and then constructs
    a result from this data.

    Parameters
    ----------
    avg_gpu_metric_values: dict
        The dict where keys are gpu based metric tags
        and values are the average values around which 
        we want data
    avg_non_gpu_metric_values: list of dict
        Per model list of: 
            keys are non gpu perf metrics, values are their 
            average values.
    value_step: int 
        The step value between two adjacent data values.
        Can be used to control the max/min of the data
        distribution in the construction result
    comparator: RunConfigResultComparator
        The comparator used to compare measurements/results
    model_name: str
        The name of the model that generated this result
    model_config: ModelConfig
        The model config used to generate this result
    """

    num_vals = 10

    # Construct a result
    run_config_result = RunConfigResult(model_name=model_name,
                                        run_config=run_config,
                                        comparator=comparator)

    # Get dict of list of metric values
    gpu_metric_values = {}
    for gpu_uuid, metric_values in avg_gpu_metric_values.items():
        gpu_metric_values[gpu_uuid] = {
            key: list(
                range(val - value_step * num_vals, val + value_step * num_vals,
                      value_step)) for key, val in metric_values.items()
        }

    non_gpu_metric_values_list = []
    for avg_non_gpu_metric_values in avg_non_gpu_metric_values_list:
        non_gpu_metric_values_list.append({
            key: list(
                range(val - value_step * num_vals, val + value_step * num_vals,
                      value_step))
            for key, val in avg_non_gpu_metric_values.items()
        })

    # Construct measurements and add them to the result
    for i in range(2 * num_vals):
        gpu_metrics = {}
        for gpu_uuid, metric_values in gpu_metric_values.items():
            gpu_metrics[gpu_uuid] = {
                key: metric_values[key][i] for key in metric_values
            }

    non_gpu_metrics = []
    for non_gpu_metric_values in non_gpu_metric_values_list:
        for i in range(2 * num_vals):
            non_gpu_metrics.append({
                key: non_gpu_metric_values[key][i]
                for key in non_gpu_metric_values
            })

        # TODO-TMA-571: Needs enhancement to support model_config_weights
        run_config_result.add_run_config_measurement(
            construct_run_config_measurement(
                model_name=model_name,
                model_config_names=[model_name],
                model_specific_pa_params=[{
                    'batch_size': 1,
                    'concurrency': 1
                }],
                gpu_metric_values=gpu_metrics,
                non_gpu_metric_values=non_gpu_metrics,
                metric_objectives=comparator._metric_weights,
                model_config_weights=[1]))

    return run_config_result


def default_encode(obj):
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        return obj.__dict__
