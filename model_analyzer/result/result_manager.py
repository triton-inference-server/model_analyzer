# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Union, DefaultDict

from model_analyzer.result.result_statistics import ResultStatistics
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.constants import TOP_MODELS_REPORT_KEY
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from .sorted_results import SortedResults
from .run_config_result_comparator import RunConfigResultComparator
from .run_config_measurement import RunConfigMeasurement
from .run_config_result import RunConfigResult
from .results import Results

from model_analyzer.config.generate.base_model_config_generator import BaseModelConfigGenerator

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_report import ConfigCommandReport
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.result.constraint_manager import ConstraintManager

from collections import defaultdict


class ResultManager:
    """
    This class provides methods to create to hold
    and sort results
    """

    def __init__(self, config: Union[ConfigCommandProfile, ConfigCommandReport],
                 state_manager: AnalyzerStateManager,
                 constraint_manager: ConstraintManager):
        """
        Parameters
        ----------
        config :ConfigCommandProfile/ConfigCommandReport
            the model analyzer config
        state_manager: AnalyzerStateManager
            The object that allows control and update of state
        constraint_manager: ConstraintManager
            The object that handles processing and applying
            constraints on a given measurements
        """

        self._config = config
        self._state_manager = state_manager
        self._constraint_manager = constraint_manager

        # Data structures for sorting results
        self._per_model_sorted_results: DefaultDict[
            str, SortedResults] = defaultdict(SortedResults)
        self._across_model_sorted_results: SortedResults = SortedResults()

        if state_manager.starting_fresh_run():
            self._init_state()

        self._complete_setup()

    def get_model_names(self):
        """
        Returns a list of model names that have sorted results
        """
        return list(self._per_model_sorted_results.keys())

    def get_model_sorted_results(self, model_name):
        """
        Returns a list of sorted results for the requested model
        """
        if model_name not in self._per_model_sorted_results:
            raise TritonModelAnalyzerException(
                f"model name {model_name} not found in result manager")
        return self._per_model_sorted_results[model_name]

    def get_across_model_sorted_results(self):
        """
        Returns a list of sorted results across all models
        """
        return self._across_model_sorted_results

    def get_results(self):
        """ Returns all results (return type is Results) """
        return self._state_manager.get_state_variable('ResultManager.results')

    def get_server_only_data(self):
        """
        Returns : dict
            keys are gpu ids and values are lists of metric values        
        """
        return self._state_manager.get_state_variable(
            'ResultManager.server_only_data')

    def add_server_data(self, data):
        """
        Adds data to directly to the server only table

        Parameters
        ----------
        data : dict
            keys are gpu ids and values are lists of metric values
        """

        self._state_manager.set_state_variable('ResultManager.server_only_data',
                                               data)

    def add_run_config_measurement(
            self, run_config: RunConfig,
            run_config_measurement: RunConfigMeasurement) -> None:
        """
        Add measurement to individual result heap,
        global result heap and results class
        """
        model_name = run_config.models_name()

        run_config_result = RunConfigResult(
            model_name=model_name,
            run_config=run_config,
            comparator=self._run_comparators[model_name],
            constraint_manager=self._constraint_manager)

        run_config_measurement.set_metric_weightings(
            self._run_comparators[model_name].get_metric_weights())

        run_config_measurement.set_model_config_weighting(
            self._run_comparators[model_name].get_model_weights())

        self._add_rcm_to_results(run_config, run_config_measurement)
        run_config_result.add_run_config_measurement(run_config_measurement)

        self._per_model_sorted_results[model_name].add_result(run_config_result)
        self._across_model_sorted_results.add_result(run_config_result)

    def get_model_configs_run_config_measurements(self, model_variants_name):
        """
        Unsorted list of RunConfigMeasurements for a config

        Parameters
        ----------
        model_variants_name: str

        Returns
        -------
        (RunConfig, list of RunConfigMeasurements)
            The measurements for a particular config, in the order
            they were obtained.
        """

        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        # Name format is <base_model_name>_config_<number_or_default>
        #
        model_name = BaseModelConfigGenerator.extract_model_name_from_variant_name(
            model_variants_name)

        # Remote mode has model_name == model_config_name
        #
        if not results.contains_model(model_name):
            model_name = model_variants_name

        if results.contains_model(
                model_name) and results.contains_model_variant(
                    model_name, model_variants_name):
            return results.get_all_model_variant_measurements(
                model_name, model_variants_name)
        else:
            raise TritonModelAnalyzerException(
                f"RunConfig {model_variants_name} requested for report step but no results were found. "
                "Double check the name and ensure that this model config was actually profiled."
            )

    def top_n_results(self,
                      model_name=None,
                      n=SortedResults.GET_ALL_RESULTS,
                      include_default=False):
        """
        Parameters
        ----------
        model_name: str
            The name of the model
            for which we need the top 
            n results.
        n : int
            The number of  top results
            to retrieve. Returns all by 
            default
        include_default : bool
            If true, the model's default config results will
            be included in the returned results. In the case
            that the default isn't one of the top n results,
            then n+1 results will be returned
        Returns
        -------
        list of RunConfigResults
            The n best results for this model,
            must all be passing results
        """

        if model_name:
            results = self._per_model_sorted_results[model_name]
        else:
            results = self._across_model_sorted_results

        top_results = results.top_n_results(n)

        if include_default:
            self._add_default_to_results(model_name, top_results, results)

        return top_results

    def get_result_statistics(self):
        """
        This function computes statistics
        with results currently in the result
        manager's heap
        """

        def _update_stats(statistics, sorted_results, stats_key):
            passing_measurements = 0
            failing_measurements = 0
            total_configs = 0
            for result in sorted_results.results():
                total_configs += 1
                passing_measurements += len(result.passing_measurements())
                failing_measurements += len(result.failing_measurements())

            statistics.set_total_configurations(stats_key, total_configs)
            statistics.set_passing_measurements(stats_key, passing_measurements)
            statistics.set_failing_measurements(stats_key, failing_measurements)

        result_stats = ResultStatistics()
        for model_name, sorted_results in self._per_model_sorted_results.items(
        ):
            _update_stats(result_stats, sorted_results, model_name)

        _update_stats(result_stats, self._across_model_sorted_results,
                      TOP_MODELS_REPORT_KEY)

        return result_stats

    def _init_state(self):
        """
        Sets ResultManager object managed
        state variables in AnalyerState
        """

        self._state_manager.set_state_variable('ResultManager.results',
                                               Results())
        self._state_manager.set_state_variable('ResultManager.server_only_data',
                                               {})

    def _complete_setup(self):
        # The Report subcommand can init, but nothing needs to be done
        if isinstance(self._config, ConfigCommandProfile):
            self._complete_profile_setup()
        elif isinstance(self._config, ConfigCommandReport):
            pass
        else:
            raise TritonModelAnalyzerException(
                f"Expected config of type ConfigCommandProfile/ConfigCommandReport,"
                f" got {type(self._config)}.")

    def _complete_profile_setup(self):
        self._create_concurrent_profile_model_name()

        if self._config.run_config_profile_models_concurrently_enable:
            self._setup_for_concurrent_profile()
        else:
            self._setup_for_sequential_profile()

        self._add_results_to_heaps(suppress_warnings=True)

    def _create_concurrent_profile_model_name(self):
        profile_model_names = [
            model.model_name() for model in self._config.profile_models
        ]

        self._concurrent_profile_model_name = ','.join(profile_model_names)

    def _profiling_models_concurrently(self):
        """
        Returns
        -------
        bool: True if we are doing concurrent model profile
        """
        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        return bool(
            results.get_model_measurements_dict(
                models_name=self._concurrent_profile_model_name,
                suppress_warning=True) and len(self._config.profile_models) > 1)

    def _setup_for_concurrent_profile(self):
        self._profile_model_names = [self._concurrent_profile_model_name]

        model_objectives_list = [
            model.objectives() for model in self._config.profile_models
        ]
        model_weighting_list = [
            model.weighting() for model in self._config.profile_models
        ]

        self._run_comparators = {
            self._concurrent_profile_model_name:
                RunConfigResultComparator(
                    metric_objectives_list=model_objectives_list,
                    model_weights=model_weighting_list)
        }

    def _setup_for_sequential_profile(self):
        self._profile_model_names = [
            model.model_name() for model in self._config.profile_models
        ]

        self._run_comparators = {
            model.model_name(): RunConfigResultComparator(
                metric_objectives_list=[model.objectives()],
                model_weights=[model.weighting()])
            for model in self._config.profile_models
        }

    def _add_rcm_to_results(self, run_config, run_config_measurement):
        """
        This function adds model inference
        measurements to the required result

        Parameters
        ----------
        run_config : RunConfig
            Contains the parameters used to generate the measurment
        run_config_measurement: RunConfigMeasurement
            the measurement to be added
        """

        # Get reference to results state and modify it
        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        results.add_run_config_measurement(run_config, run_config_measurement)

        # Use set_state_variable to record that state may have been changed
        self._state_manager.set_state_variable(name='ResultManager.results',
                                               value=results)

    def _add_results_to_heaps(self, suppress_warnings=False):
        """
        Construct and add results to individual result heaps 
        as well as global result heap
        """
        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        for model_name in self._profile_model_names:
            model_measurements = results.get_model_measurements_dict(
                model_name, suppress_warnings)

            # Only add in models that exist in the checkpoint
            if not model_measurements:
                continue

            for (run_config,
                 run_config_measurements) in model_measurements.values():
                run_config_result = RunConfigResult(
                    model_name=model_name,
                    run_config=run_config,
                    comparator=self._run_comparators[model_name],
                    constraint_manager=self._constraint_manager)

                for run_config_measurement in run_config_measurements.values():
                    run_config_measurement.set_metric_weightings(
                        self._run_comparators[model_name].get_metric_weights())

                    run_config_measurement.set_model_config_weighting(
                        self._run_comparators[model_name].get_model_weights())

                    run_config_result.add_run_config_measurement(
                        run_config_measurement)

                self._per_model_sorted_results[model_name].add_result(
                    run_config_result)
                self._across_model_sorted_results.add_result(run_config_result)

    def _add_default_to_results(self, model_name, results, sorted_results):
        '''
        If default config is already in results, keep it there. Else, find and
        add it from the result heap
        '''
        if not model_name:
            return

        model_names = model_name.split(",")
        model_names = [
            model_name + "_config_default" for model_name in model_names
        ]
        default_model_name = ','.join(model_names)

        for run_config_result in results:
            if run_config_result.run_config().model_variants_name(
            ) == default_model_name:
                return

        for run_config_result in sorted_results.results():
            if run_config_result.run_config().model_variants_name(
            ) == default_model_name:
                results.append(run_config_result)
                return
