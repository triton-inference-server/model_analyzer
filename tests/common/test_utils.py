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

from model_analyzer.result.measurement import Measurement
from model_analyzer.result.model_result import ModelResult
from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig


def construct_measurement(model_name, gpu_metric_values, non_gpu_metric_values,
                          comparator):
    """
    Construct a measurement from the given data

    Parameters
    ----------
    model_name: str
        The name of the model that generated this result
    gpu_metric_values: dict
        Keys are gpu id, values are dict
        The dict where keys are gpu based metric tags, values are the data
    non_gpu_metric_values: dict
        Keys are non gpu perf metrics, values are their values
    comparator: ResultComparator
        The comparator used to compare measurements/results
    
    Returns
    -------
    Measurement
        constructed with all of the above data.
    """

    # gpu_data will be a dict whose keys are gpu_ids and values
    # are lists of Records
    gpu_data = {}
    for gpu_uuid, metrics_values in gpu_metric_values.items():
        gpu_data[gpu_uuid] = []
        gpu_metric_tags = list(metrics_values.keys())
        for i, gpu_metric in enumerate(
                MetricsManager.get_metric_types(gpu_metric_tags)):
            gpu_data[gpu_uuid].append(
                gpu_metric(value=metrics_values[gpu_metric_tags[i]]))

    # Non gpu data will be a list of records
    non_gpu_data = []
    non_gpu_metric_tags = list(non_gpu_metric_values.keys())
    for i, metric in enumerate(
            MetricsManager.get_metric_types(non_gpu_metric_tags)):
        non_gpu_data.append(
            metric(value=non_gpu_metric_values[non_gpu_metric_tags[i]]))

    # Perf Config needs a protocol
    perf_config = PerfAnalyzerConfig()
    perf_config['model-name'] = model_name
    perf_config['protocol'] = 'http'

    measurement = Measurement(gpu_data=gpu_data,
                              non_gpu_data=non_gpu_data,
                              perf_config=perf_config)
    measurement.set_result_comparator(comparator=comparator)
    return measurement


def construct_result(avg_gpu_metric_values,
                     avg_non_gpu_metric_values,
                     comparator,
                     value_step=1,
                     model_name=None,
                     model_config=None):
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
    avg_non_gpu_metric_values: dict
        Keys are non gpu perf metrics, values are their 
        average values.
    value_step: int 
        The step value between two adjacent data values.
        Can be used to control the max/min of the data
        distribution in the construction result
    comparator: ResultComparator
        The comparator used to compare measurements/results
    model_name: str
        The name of the model that generated this result
    model_config: ModelConfig
        The model config used to generate this result
    """

    num_vals = 10

    # Construct a result
    model_result = ModelResult(model_name=model_name,
                               model_config=model_config,
                               comparator=comparator)

    # Get dict of list of metric values
    gpu_metric_values = {}
    for gpu_uuid, metric_values in avg_gpu_metric_values.items():
        gpu_metric_values[gpu_uuid] = {
            key: list(
                range(val - value_step * num_vals, val + value_step * num_vals,
                      value_step)) for key, val in metric_values.items()
        }

    non_gpu_metric_values = {
        key: list(
            range(val - value_step * num_vals, val + value_step * num_vals,
                  value_step))
        for key, val in avg_non_gpu_metric_values.items()
    }

    # Construct measurements and add them to the result
    for i in range(2 * num_vals):
        gpu_metrics = {}
        for gpu_uuid, metric_values in gpu_metric_values.items():
            gpu_metrics[gpu_uuid] = {
                key: metric_values[key][i] for key in metric_values
            }
        non_gpu_metrics = {
            key: non_gpu_metric_values[key][i] for key in non_gpu_metric_values
        }
        model_result.add_measurement(
            construct_measurement(model_name=model_name,
                                  gpu_metric_values=gpu_metrics,
                                  non_gpu_metric_values=non_gpu_metrics,
                                  comparator=comparator))

    return model_result
