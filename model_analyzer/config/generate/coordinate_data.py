# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Tuple
from model_analyzer.config.generate.coordinate import Coordinate
from model_analyzer.result.run_config_measurement import RunConfigMeasurement


class CoordinateData:
    """
    A class that tracks the measurement data in the current neighborhood
    and the visit counts of all the coordinates in the coordinate space.
    """

    def __init__(self):
        self._measurements = {}
        self._visit_counts = {}

    def get_measurement(self, coordinate: Coordinate) -> RunConfigMeasurement:
        """
        Return the measurement data of the given coordinate.
        """
        key: Tuple[Coordinate, ...] = tuple(coordinate)
        return self._measurements.get(key, None)

    def set_measurement(self,
                        coordinate: Coordinate,
                        measurement: RunConfigMeasurement):
        """
        Set the measurement for the given coordinate.
        """
        key: Tuple[Coordinate, ...] = tuple(coordinate)
        self._measurements[key] = measurement

    def reset_measurements(self):
        """
        Resets the collection of measurements.
        """
        self._measurements = {}

    def get_visit_count(self, coordinate: Coordinate) -> int:
        """
        Get the visit count for the given coordinate. 
        Returns 0 if the coordinate hasn't been visited yet
        """
        key: Tuple[Coordinate, ...] = tuple(coordinate)
        return self._visit_counts.get(key, 0)

    def increment_visit_count(self, coordinate: Coordinate):
        """
        Increase the visit count for the given coordinate by 1
        """
        key: Tuple[Coordinate, ...] = tuple(coordinate)
        new_count = self.get_visit_count(coordinate) + 1
        self._visit_counts[key] = new_count
