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

from typing import Dict, List, Union, Optional, Generator, Tuple

from .config_generator_interface import ConfigGeneratorInterface

from model_analyzer.config.generate.base_model_config_generator import BaseModelConfigGenerator
from model_analyzer.config.generate.search_config import SearchConfig
from model_analyzer.config.generate.coordinate import Coordinate
from model_analyzer.config.generate.coordinate_data import CoordinateData
from model_analyzer.config.generate.neighborhood import Neighborhood
from model_analyzer.config.generate.brute_run_config_generator import BruteRunConfigGenerator
from model_analyzer.config.generate.model_variant_name_manager import ModelVariantNameManager
from model_analyzer.config.generate.model_profile_spec import ModelProfileSpec
from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.triton.client.client import TritonClient
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

from sys import maxsize

from model_analyzer.constants import LOGGER_NAME

import logging

logger = logging.getLogger(LOGGER_NAME)


class QuickRunConfigGenerator(ConfigGeneratorInterface):
    """
    Hill climbing algorithm to create RunConfigs
    """

    def __init__(self, search_config: SearchConfig,
                 config: ConfigCommandProfile, gpus: List[GPUDevice],
                 models: List[ModelProfileSpec],
                 ensemble_composing_models: Dict[str, List[ModelProfileSpec]],
                 bls_composing_models: List[ModelProfileSpec],
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
        models: List of ModelProfileSpec
            List of models to profile
        ensemble_composing_models: Dict of List of ModelProfileSpec
            Dict indexed by model name of ensemble composing model profiles
        bls_composing_models: List of ModelProfileSpec
            List of BLS composing model profiles
        client: TritonClient
        model_variant_name_manager: ModelVariantNameManager
        """
        self._search_config = search_config
        self._config = config
        self._client = client
        self._gpus = gpus
        self._models = models
        self._ensemble_composing_models = ensemble_composing_models
        self._bls_composing_models = bls_composing_models

        self._model_variant_name_manager = model_variant_name_manager

        self._triton_env = BruteRunConfigGenerator.determine_triton_server_env(
            models)

        # This tracks measured results for all coordinates
        self._coordinate_data = CoordinateData()

        # This is an initial center that the neighborhood is built around.
        # It is updated every new creation of the neighborhood.
        self._home_coordinate = self._get_starting_coordinate()

        # This is the coordinate that we want to measure next. It is
        # updated every step of this generator
        self._coordinate_to_measure: Coordinate = self._home_coordinate

        # Track the best coordinate seen so far that can be used during
        # the back-off stage.
        self._best_coordinate = self._home_coordinate
        self._best_measurement: Optional[RunConfigMeasurement] = None

        self._neighborhood = Neighborhood(
            self._search_config.get_neighborhood_config(),
            self._home_coordinate, self._coordinate_data)

        # Sticky bit. Once true, we should never stay at a home that is failing or None
        self._home_has_passed = False

        self._done = False

    def _is_done(self) -> bool:
        return self._done

    def get_configs(self) -> Generator[RunConfig, None, None]:
        """
        Returns
        -------
        RunConfig
            The next RunConfig generated by this class
        """
        config = self._create_default_run_config()
        yield (config)

        while True:
            if self._is_done():
                break

            config = self._get_next_run_config()
            yield (config)
            self._step()

    def _step(self) -> None:
        """
        Determine self._coordinate_to_measure, which is what is used to
        create the next RunConfig
        """
        if self._should_step_back():
            self._take_step_back()
        elif self._neighborhood.enough_coordinates_initialized():
            self._take_step()
        else:
            self._pick_coordinate_to_initialize()

    def set_last_results(
            self, measurements: List[Optional[RunConfigMeasurement]]) -> None:
        """
        Given the results from the last RunConfig, make decisions
        about future configurations to generate

        Parameters
        ----------
        measurements: List of Measurements from the last run(s)
        """
        self._coordinate_data.set_measurement(
            coordinate=self._coordinate_to_measure, measurement=measurements[0])

        if measurements[0] is not None:
            self._update_best_measurement(measurement=measurements[0])

            if self._measuring_home_coordinate(
            ) and measurements[0].is_passing_constraints():
                self._home_has_passed = True

        self._print_debug_logs(measurements)

    def _update_best_measurement(self,
                                 measurement: RunConfigMeasurement) -> None:
        """Keep track of the best coordinate/measurement seen so far."""
        if self._best_measurement is None:
            self._best_coordinate = self._coordinate_to_measure
            self._best_measurement = measurement

        elif not self._best_measurement.is_passing_constraints(
        ) and measurement.is_passing_constraints():
            self._best_coordinate = self._coordinate_to_measure
            self._best_measurement = measurement

        elif not self._best_measurement.is_passing_constraints(
        ) and not measurement.is_passing_constraints():
            comparison = self._best_measurement.compare_constraints(
                other=measurement)

            if comparison and comparison > 0:
                self._best_coordinate = self._coordinate_to_measure
                self._best_measurement = measurement

        elif self._best_measurement.is_passing_constraints(
        ) and measurement.is_passing_constraints():
            comparison = self._best_measurement.compare_measurements(
                other=measurement)

            if comparison and comparison > 0:
                self._best_coordinate = self._coordinate_to_measure
                self._best_measurement = measurement

    def _get_last_results(self) -> Optional[RunConfigMeasurement]:
        return self._coordinate_data.get_measurement(
            coordinate=self._coordinate_to_measure)

    def _take_step(self) -> None:
        new_coordinate = self._neighborhood.determine_new_home()
        self._determine_if_done(new_coordinate)

        logger.debug(f"Stepping {self._home_coordinate}->{new_coordinate}")
        self._home_coordinate = new_coordinate
        self._coordinate_to_measure = new_coordinate
        self._recreate_neighborhood(force_slow_mode=False)

    def _take_step_back(self) -> None:
        new_coordinate = self._neighborhood.get_nearest_neighbor(
            coordinate_in=self._best_coordinate)

        # TODO: TMA-871: handle back-off (and its termination) better.
        if new_coordinate == self._home_coordinate:
            self._done = True

        logger.debug(
            f"Stepping back: {self._home_coordinate}->{new_coordinate}")
        self._home_coordinate = new_coordinate
        self._coordinate_to_measure = new_coordinate
        self._recreate_neighborhood(force_slow_mode=True)

    def _should_step_back(self) -> bool:
        """
        Step back if take any of the following steps:
          - Step from a passing home to a failing home
          - Step from any home to home with a None measurement
        """
        if self._measuring_home_coordinate():
            last_results = self._get_last_results()
            if not last_results:
                return True
            last_results_passed = last_results.is_passing_constraints()
            if not last_results_passed and self._home_has_passed:
                return True
        return False

    def _measuring_home_coordinate(self) -> bool:
        return self._coordinate_to_measure == self._home_coordinate

    def _determine_if_done(self, new_coordinate: Coordinate) -> None:
        """
        Based on the new coordinate picked, determine if the generator is done
        and if so, update self._done
        """
        if new_coordinate == self._home_coordinate:
            self._done = True
        if self._coordinate_data.get_visit_count(new_coordinate) >= 2:
            self._done = True

    def _recreate_neighborhood(self, force_slow_mode: bool) -> None:
        neighborhood_config = self._search_config.get_neighborhood_config()

        self._neighborhood = Neighborhood(neighborhood_config,
                                          self._home_coordinate,
                                          self._coordinate_data)

        self._coordinate_data.increment_visit_count(self._home_coordinate)

        if force_slow_mode:
            self._neighborhood.force_slow_mode()

    def _pick_coordinate_to_initialize(self) -> None:
        next_coordinate = self._neighborhood.pick_coordinate_to_initialize()

        if next_coordinate:
            self._coordinate_to_measure = next_coordinate
            logger.debug(
                f"Need more data. Measuring {self._coordinate_to_measure}")
        else:
            logger.info("No coordinate to measure. Exiting")
            self._done = True

    def _get_starting_coordinate(self) -> Coordinate:
        min_indexes = self._search_config.get_min_indexes()
        return Coordinate(min_indexes)

    def _get_coordinate_values(self, coordinate: Coordinate,
                               key: int) -> Dict[str, Union[int, float]]:
        dims = self._search_config.get_dimensions()
        values = dims.get_values_for_coordinate(coordinate)
        return values[key]

    def _get_next_run_config(self) -> RunConfig:
        run_config = RunConfig(self._triton_env)

        model_index = 0
        for model in self._models:
            mrc, model_index = self._get_next_model_run_config(
                model, model_index)
            run_config.add_model_run_config(mrc)

        return run_config

    def _get_next_model_run_config(
            self, model: ModelProfileSpec,
            model_index: int) -> Tuple[ModelRunConfig, int]:
        start_model_index = model_index

        if model.model_name() in self._ensemble_composing_models:
            ensemble_composing_configs, end_model_index = self._get_next_ensemble_composing_configs(
                model, start_model_index)

            param_combo = self._get_next_ensemble_param_combo(
                start_model_index, end_model_index)

            model_config = self._get_next_ensemble_model_config(
                model, ensemble_composing_configs, param_combo)

            model_index = end_model_index
        else:
            model_config = self._get_next_model_config(model, model_index)
            model_index = model_index + 1

        model_variant_name = model_config.get_field('name')
        perf_analyzer_config = self._get_next_perf_analyzer_config(
            model_variant_name, model, start_model_index)

        model_name = model.model_name()
        model_run_config = ModelRunConfig(model_name, model_config,
                                          perf_analyzer_config)

        if model.model_name() in self._ensemble_composing_models:
            model_run_config.add_ensemble_composing_model_configs(
                ensemble_composing_configs)

        return (model_run_config, model_index)

    def _get_next_ensemble_composing_configs(
            self, model: ModelProfileSpec,
            start_model_index: int) -> Tuple[List[ModelConfig], int]:
        model_index = start_model_index
        ensemble_composing_configs = []
        for ensemble_composing_model in self._ensemble_composing_models[
                model.model_name()]:
            ensemble_composing_configs.append(
                self._get_next_model_config(ensemble_composing_model,
                                            model_index))
            model_index = model_index + 1

        return (ensemble_composing_configs, model_index)

    def _get_next_ensemble_param_combo(self, start_model_index: int,
                                       end_model_index: int) -> dict:
        """
        For the ensemble model the only parameter we need to set 
        is the max batch size; which will be the minimum batch size 
        found in the composing_model max batch sizes
        """
        min_val_of_max_batch_size = maxsize
        for model_index in range(start_model_index, end_model_index):
            dimension_values = self._get_coordinate_values(
                self._coordinate_to_measure, model_index)

            min_val_of_max_batch_size = int(
                min([
                    dimension_values.get("max_batch_size", 1),
                    min_val_of_max_batch_size
                ]))

        param_combo = {'max_batch_size': min_val_of_max_batch_size}

        return param_combo

    def _get_next_ensemble_model_config(
            self, model: ModelProfileSpec,
            ensemble_composing_configs: List[ModelConfig],
            param_combo: dict) -> ModelConfig:
        model_config = BaseModelConfigGenerator.make_ensemble_model_config(
            model=model,
            ensemble_composing_model_configs=ensemble_composing_configs,
            model_variant_name_manager=self._model_variant_name_manager,
            param_combo=param_combo)

        return model_config

    def _get_next_model_config(self, model: ModelProfileSpec,
                               dimension_index: int) -> ModelConfig:
        dimension_values = self._get_coordinate_values(
            self._coordinate_to_measure, dimension_index)

        kind = "KIND_CPU" if model.cpu_only() else "KIND_GPU"
        instance_count = self._calculate_instance_count(dimension_values)

        param_combo: dict = {
            'instance_group': [{
                'count': instance_count,
                'kind': kind,
            }]
        }

        if 'max_batch_size' in dimension_values:
            param_combo['max_batch_size'] = self._calculate_model_batch_size(
                dimension_values)

        if model.supports_dynamic_batching():
            param_combo['dynamic_batching'] = {}

        model_config = BaseModelConfigGenerator.make_model_config(
            param_combo=param_combo,
            model=model,
            model_variant_name_manager=self._model_variant_name_manager)
        return model_config

    def _get_next_perf_analyzer_config(self, model_variant_name: str,
                                       model: ModelProfileSpec,
                                       model_index: int) -> PerfAnalyzerConfig:
        dimension_values = self._get_coordinate_values(
            self._coordinate_to_measure, model_index)

        perf_analyzer_config = PerfAnalyzerConfig()

        perf_analyzer_config.update_config_from_profile_config(
            model_variant_name, self._config)

        concurrency = self._calculate_concurrency(dimension_values)

        perf_config_params = {'batch-size': 1, 'concurrency-range': concurrency}
        perf_analyzer_config.update_config(perf_config_params)

        perf_analyzer_config.update_config(model.perf_analyzer_flags())
        return perf_analyzer_config

    def _calculate_model_batch_size(
            self, dimension_values: Dict[str, Union[int, float]]) -> int:
        batch_size = int(dimension_values.get("max_batch_size", 1))

        min_batch_size_is_set_by_config = self._config.get_config(
        )['run_config_search_min_model_batch_size'].is_set_by_config()

        max_batch_size_is_set_by_config = self._config.get_config(
        )['run_config_search_max_model_batch_size'].is_set_by_config()

        if min_batch_size_is_set_by_config and batch_size < self._config.run_config_search_min_model_batch_size:
            return self._config.run_config_search_min_model_batch_size

        if max_batch_size_is_set_by_config and batch_size > self._config.run_config_search_max_model_batch_size:
            return self._config.run_config_search_max_model_batch_size

        return batch_size

    def _calculate_instance_count(
            self, dimension_values: Dict[str, Union[int, float]]) -> int:
        instance_count = int(dimension_values.get("instance_count", 1))

        min_instance_count_is_set_by_config = self._config.get_config(
        )['run_config_search_min_instance_count'].is_set_by_config()

        max_instance_count_is_set_by_config = self._config.get_config(
        )['run_config_search_max_instance_count'].is_set_by_config()

        if min_instance_count_is_set_by_config and instance_count < self._config.run_config_search_min_instance_count:
            return self._config.run_config_search_min_instance_count

        if max_instance_count_is_set_by_config and instance_count > self._config.run_config_search_max_instance_count:
            return self._config.run_config_search_max_instance_count

        return instance_count

    def _calculate_concurrency(
            self, dimension_values: Dict[str, Union[int, float]]) -> int:
        model_batch_size = self._calculate_model_batch_size(dimension_values)
        instance_count = self._calculate_instance_count(dimension_values)
        concurrency = 2 * model_batch_size * instance_count

        min_concurrency_is_set_by_config = self._config.get_config(
        )['run_config_search_min_concurrency'].is_set_by_config()

        max_concurrency_is_set_by_config = self._config.get_config(
        )['run_config_search_max_concurrency'].is_set_by_config()

        if min_concurrency_is_set_by_config and concurrency < self._config.run_config_search_min_concurrency:
            return self._config.run_config_search_min_concurrency

        if max_concurrency_is_set_by_config and concurrency > self._config.run_config_search_max_concurrency:
            return self._config.run_config_search_max_concurrency

        return concurrency

    def _create_default_run_config(self) -> RunConfig:
        default_run_config = RunConfig(self._triton_env)

        for model in self._models:
            if model.model_name() in self._ensemble_composing_models:
                default_run_config.add_model_run_config(
                    self._create_default_ensemble_model_run_config(model))
            else:
                default_run_config.add_model_run_config(
                    self._create_default_model_run_config(model))

        return default_run_config

    def _create_default_ensemble_model_run_config(
            self, model: ModelProfileSpec) -> ModelRunConfig:
        default_composing_model_configs = self._create_default_composing_model_configs(
            model)

        default_ensemble_model_config = BaseModelConfigGenerator.make_ensemble_model_config(
            model=model,
            ensemble_composing_model_configs=default_composing_model_configs,
            model_variant_name_manager=self._model_variant_name_manager)

        default_perf_analyzer_config = self._create_default_perf_analyzer_config(
            model, default_ensemble_model_config)

        default_model_run_config = ModelRunConfig(
            model.model_name(), default_ensemble_model_config,
            default_perf_analyzer_config)

        default_model_run_config.add_ensemble_composing_model_configs(
            default_composing_model_configs)

        return default_model_run_config

    def _create_default_composing_model_configs(
            self, model: ModelProfileSpec) -> List[ModelConfig]:
        default_composing_model_configs: List[ModelConfig] = []
        if self._ensemble_composing_models:
            for composing_model in self._ensemble_composing_models[
                    model.model_name()]:
                default_composing_model_configs.append(
                    BaseModelConfigGenerator.make_model_config(
                        param_combo={},
                        model=composing_model,
                        model_variant_name_manager=self.
                        _model_variant_name_manager))

        for composing_model in self._bls_composing_models:
            default_composing_model_configs.append(
                BaseModelConfigGenerator.make_model_config(
                    param_combo={},
                    model=composing_model,
                    model_variant_name_manager=self._model_variant_name_manager)
            )

        return default_composing_model_configs

    def _create_default_model_run_config(
            self, model: ModelProfileSpec) -> ModelRunConfig:
        default_model_config = BaseModelConfigGenerator.make_model_config(
            param_combo={},
            model=model,
            model_variant_name_manager=self._model_variant_name_manager)

        default_perf_analyzer_config = self._create_default_perf_analyzer_config(
            model, default_model_config)

        default_model_run_config = ModelRunConfig(model.model_name(),
                                                  default_model_config,
                                                  default_perf_analyzer_config)

        default_composing_model_configs = self._create_default_composing_model_configs(
            model)

        if default_composing_model_configs:
            default_model_run_config.add_bls_composing_model_configs(
                default_composing_model_configs)

        return default_model_run_config

    def _create_default_perf_analyzer_config(
            self, model: ModelProfileSpec,
            model_config: ModelConfig) -> PerfAnalyzerConfig:
        default_perf_analyzer_config = PerfAnalyzerConfig()
        default_perf_analyzer_config.update_config_from_profile_config(
            model_config.get_field('name'), self._config)

        perf_config_params = {'batch-size': 1, 'concurrency-range': 1}
        default_perf_analyzer_config.update_config(perf_config_params)

        default_perf_analyzer_config.update_config(model.perf_analyzer_flags())

        return default_perf_analyzer_config

    def _print_debug_logs(
            self, measurements: List[Union[RunConfigMeasurement,
                                           None]]) -> None:
        if measurements is not None and measurements[0] is not None:
            assert len(measurements) == 1

            throughput = measurements[0].get_non_gpu_metric_value(
                "perf_throughput")
            latency = measurements[0].get_non_gpu_metric_value(
                "perf_latency_p99")

            if self._best_measurement:
                best_throughput = self._best_measurement.get_non_gpu_metric_value(
                    "perf_throughput")
                best_latency = self._best_measurement.get_non_gpu_metric_value(
                    "perf_latency_p99")
            else:
                best_throughput = 0
                best_latency = 0

            logger.debug(
                f"Measurement for {self._coordinate_to_measure}: "
                f"throughput = {throughput}, latency = {latency} "
                f"(best throughput: {best_throughput}, best_latency: {best_latency})"
            )
        else:
            logger.debug(
                f"Measurement for {self._coordinate_to_measure}: None.")
