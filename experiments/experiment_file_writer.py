# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import csv


class ExperimentFileWriter:
    """
    A file writer that collects all the results from the undirected algorithm
    and the corresponding brute-force algorithm and writes them into csv files.
    """
    field_names = [
        "overall_num_measurements", "overall_best_throughput",
        "undirected_num_measurements", "missing_num_measurements",
        "undirected_throughput", "radius", "magnitude", "min_initialized"
    ]

    def __init__(self, output_path, file_name="output_vgg19_libtorch.csv"):
        self._filename = os.path.join(output_path, file_name)

        if not os.path.exists(self._filename):
            os.makedirs(output_path, exist_ok=True)

            with open(self._filename, mode="w") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
                writer.writeheader()

    def write(self, checkpoint_data, profile_data, radius, magnitude,
              min_initialized):
        try:
            with open(self._filename, mode="a") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.field_names)

                overall_best_measurement = checkpoint_data.get_best_run_config_measurement(
                )
                undirected_best_measurement = profile_data.get_best_run_config_measurement(
                )

                # yapf: disable
                writer.writerow({
                    "overall_num_measurements":
                        checkpoint_data.get_run_config_measurement_count(),
                    "overall_best_throughput":
                        overall_best_measurement.get_non_gpu_metric_value("perf_throughput"),
                    "undirected_num_measurements":
                        profile_data.get_run_config_measurement_count(),
                    "missing_num_measurements":
                        checkpoint_data.get_missing_measurement_count(),
                    "undirected_throughput":
                        undirected_best_measurement.get_non_gpu_metric_value("perf_throughput"),
                    "radius": radius,
                    "magnitude": magnitude,
                    "min_initialized": min_initialized
                })
                # yapf: enable
        except OSError as e:
            raise TritonModelAnalyzerException(e)
