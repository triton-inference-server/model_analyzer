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

from typing import List

from .config_generator_interface import ConfigGeneratorInterface

from model_analyzer.config.generate.base_model_config_generator import BaseModelConfigGenerator
from model_analyzer.config.generate.search_config import SearchConfig
from model_analyzer.config.generate.coordinate import Coordinate
from model_analyzer.config.generate.coordinate_data import CoordinateData
from model_analyzer.config.generate.neighborhood import Neighborhood
from model_analyzer.config.generate.brute_run_config_generator import BruteRunConfigGenerator
from model_analyzer.config.generate.model_variant_name_manager import ModelVariantNameManager
from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.objects.config_model_profile_spec import ConfigModelProfileSpec
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

import logging

logger = logging.getLogger(LOGGER_NAME)


class QuickRunConfigGenerator(ConfigGeneratorInterface):
    """
    Hill climbing algorithm to create RunConfigs
    """

    def __init__(self,
                 search_config: SearchConfig,
                 config: ConfigCommandProfile,
                 gpus: List[GPUDevice],
                 models: List[ConfigModelProfileSpec],
                 client: TritonClient,
                 model_variant_name_manager: ModelVariantNameManager):
        """
        Parameters
        ----------
        search_config: SearchConfig
            Defines parameters and dimensions for the search
        config: ConfigCommandProfile
            Profile configuration information
        gpus: List of GPUDevices
        models: List of ConfigModelProfileSpec
            List of models to profile
        client: TritonClient
        model_variant_name_manager: ModelVariantNameManager
        """
        self._search_config = search_config
        self._config = config
        self._gpus = gpus
        self._models = models
        self._client = client
        self._model_variant_name_manager = model_variant_name_manager

        self._triton_env = BruteRunConfigGenerator.determine_triton_server_env(
            models)

        # This tracks measured results for all coordinates
        self._coordinate_data = CoordinateData()

        # This is our current location that the neighborhood is built around
        self._current_coordinate = self._get_starting_coordinate()

        # This is the coordinate that we want to measure next. It is
        # updated every step of this generator
        self._coordinate_to_measure = self._current_coordinate

        # TODO: Add cases to use these
        self._radius_offset = 0
        self._magnitude_offset = 0

        self._neighborhood = Neighborhood(
            self._search_config.get_neighborhood_config(),
            self._coordinate_data, self._current_coordinate)

        self._done = False

    def _is_done(self) -> bool:
        return self._done

    def get_configs(self) -> RunConfig:
        """
        Returns
        -------
        RunConfig
            The next RunConfig generated by this class
        """

        while True:
            if self._is_done():
                break

            config = self._get_next_run_config()
            yield (config)
            self._step()

    def _step(self):
        """
        Determine self._coordinate_to_measure, which is what is used to
        create the next RunConfig
        """
        if self._neighborhood.enough_coordinates_initialized():
            self._take_step()
        else:
            self._pick_coordinate_to_initialize()

        if self._coordinate_to_measure is None:
            logger.info("No coordinate to measure. Exiting")
            self._done = True

    def set_last_results(self, measurements: List[RunConfigMeasurement]):
        """
        Given the results from the last RunConfig, make decisions
        about future configurations to generate

        Parameters
        ----------
        measurements: List of Measurements from the last run(s)
        """

        self._coordinate_data.increment_visit_count(self._coordinate_to_measure)

        if measurements is not None and measurements[0] is not None:
            assert len(measurements) == 1, \
                "Multi-model is currently not supported in quick search mode."

            # TODO: Remove.
            throughput = measurements[0].get_non_gpu_metric_value(
                "perf_throughput")
            avg_latency = measurements[0].get_non_gpu_metric_value(
                "perf_latency_avg")

            self._coordinate_data.set_measurement(
                coordinate=self._coordinate_to_measure, measurement=measurements[0])
            logger.debug(
                f"Measurement for {self._coordinate_to_measure}: "
                f"throughput = {throughput}, avg_latency = {avg_latency}"
            )
        else:
            # TODO: How to handle this case?
            #  1. throughput = 0 and latency = ?
            #       => how to compute metric & weighed vectors?
            #  2. Just treat it None
            #       => should we consider it initialized?
            self._coordinate_data.set_measurement(
                coordinate=self._coordinate_to_measure, measurement=None)
            logger.debug(
                f"Measurement for {self._coordinate_to_measure}: None"
            )

    def _get_last_results(self) -> RunConfigMeasurement:
        return self._coordinate_data.get_measurement(
            coordinate=self._coordinate_to_measure)

    def _take_step(self):
        magnitude = self._get_magnitude()

        new_coordinate = self._neighborhood.calculate_new_coordinate(magnitude)
        self._determine_if_done(new_coordinate)

        logger.debug(f"Stepping {self._current_coordinate}->{new_coordinate}")
        self._current_coordinate = new_coordinate
        self._coordinate_to_measure = new_coordinate
        self._recreate_neighborhood()

    def _determine_if_done(self, new_coordinate: Coordinate):
        """
        Based on the new coordinate picked, determine if the generator is done
        and if so, update self._done
        """
        if new_coordinate == self._current_coordinate:
            self._done = True
        if self._coordinate_data.get_visit_count(new_coordinate) >= 2:
            self._done = True

    def _recreate_neighborhood(self):
        neighborhood_config = self._search_config.get_neighborhood_config(
            self._get_radius())

        self._coordinate_data.reset_measurements()
        self._neighborhood = Neighborhood(neighborhood_config,
                                          self._coordinate_data,
                                          self._current_coordinate)

    def _pick_coordinate_to_initialize(self):
        self._coordinate_to_measure = self._neighborhood.pick_coordinate_to_initialize(
        )
        logger.debug(f"Need more data. Measuring {self._coordinate_to_measure}")

    def _get_starting_coordinate(self) -> Coordinate:
        min_indexes = self._search_config.get_min_indexes()
        return Coordinate(min_indexes)

    def _get_coordinate_values(self,
                               coordinate: Coordinate,
                               key: int) -> int:
        dims = self._search_config.get_dimensions()
        values = dims.get_values_for_coordinate(coordinate)
        return values[key]

    def _get_radius(self) -> int:
        return self._search_config.get_radius() + self._radius_offset

    def _get_magnitude(self) -> int:
        return self._search_config.get_step_magnitude() + self._magnitude_offset

    def _get_next_run_config(self) -> RunConfig:
        run_config = RunConfig(self._triton_env)

        for i, _ in enumerate(self._models):
            mrc = self._get_next_model_run_config(i)
            run_config.add_model_run_config(mrc)

        return run_config

    def _get_next_model_run_config(self, model_num: int) -> ModelRunConfig:
        mc = self._get_next_model_config(model_num)

        model_variant_name = mc.get_field('name')
        pac = self._get_next_perf_analyzer_config(model_variant_name, model_num)

        model_name = self._models[model_num].model_name()
        return ModelRunConfig(model_name, mc, pac)

    def _get_next_model_config(self, model_num: int) -> ModelConfig:
        dimension_values = self._get_coordinate_values(
            self._coordinate_to_measure, model_num)

        param_combo = {
            'dynamic_batching': {},
            'max_batch_size':
                dimension_values['max_batch_size'],
            'instance_group': [{
                'count': dimension_values['instance_count'],
                'kind': "KIND_GPU",
                'rate_limiter': {
                    'priority': 1
                }
            }]
        }

        model_config = BaseModelConfigGenerator.make_model_config(
            param_combo=param_combo,
            config=self._config,
            client=self._client,
            gpus=self._gpus,
            model=self._models[model_num],
            model_repository=self._config.model_repository,
            model_variant_name_manager=self._model_variant_name_manager)
        return model_config

    def _get_next_perf_analyzer_config(self,
                                       model_variant_name: str,
                                       model_num: int) -> PerfAnalyzerConfig:
        dimension_values = self._get_coordinate_values(
            self._coordinate_to_measure, model_num)

        perf_analyzer_config = PerfAnalyzerConfig()

        perf_analyzer_config.update_config_from_profile_config(
            model_variant_name, self._config)

        perf_config_params = {
            'batch-size':
                1,
            'concurrency-range':
                2 * dimension_values['instance_count'] *
                dimension_values['max_batch_size']
        }
        perf_analyzer_config.update_config(perf_config_params)

        perf_analyzer_config.update_config(
            self._models[model_num].perf_analyzer_flags())
        return perf_analyzer_config
