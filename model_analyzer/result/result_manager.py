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

from model_analyzer.result.result_statistics import ResultStatistics
from model_analyzer.constants import TOP_MODELS_REPORT_KEY
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from .result_heap import ResultHeap
from .run_config_result_comparator import RunConfigResultComparator
from .run_config_result import RunConfigResult
from .results import Results

from collections import defaultdict


class ResultManager:
    """
    This class provides methods to create to hold
    and sort results
    """

    def __init__(self, config, state_manager):
        """
        Parameters
        ----------
        config :ConfigCommandProfile
            the model analyzer config
        state_manager: AnalyzerStateManager
            The object that allows control and update of state
        """

        self._config = config
        self._state_manager = state_manager

        if state_manager.starting_fresh_run():
            self._init_state()

        # Data structures for sorting results
        self._per_model_sorted_results = defaultdict(ResultHeap)
        self._across_model_sorted_results = ResultHeap()

    def get_per_model_sorted_results(self):
        return self._per_model_sorted_results

    def get_across_model_sorted_results(self):
        return self._across_model_sorted_results

    def get_results(self):
        return self._state_manager.get_state_variable('ResultManager.results')

    def get_server_only_data(self):
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

    def add_run_config_measurement(self, run_config, run_config_measurement):
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

    def compile_and_sort_results(self):
        """
        Collects objectives and constraints for
        each model, constructs results from the
        measurements obtained, and sorts and 
        filters them according to constraints
        and objectives.
        """

        self._create_concurrent_analysis_model_name()

        if self._analyzing_models_concurrently():
            self._setup_for_concurrent_analysis()
        else:
            self._setup_for_sequential_analysis()

        self._add_results_to_heaps()

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
        model_name = model_variants_name.rsplit('_', 2)[0]

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

    def top_n_results(self, model_name=None, n=-1, include_default=False):
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
            result_heap = self._per_model_sorted_results[model_name]
        else:
            result_heap = self._across_model_sorted_results
        results = result_heap.top_n_results(n)

        if include_default:
            self._add_default_to_results(model_name, results, result_heap)

        return results

    def get_result_statistics(self):
        """
        This function computes statistics
        with results currently in the result
        manager's heap
        """

        def _update_stats(statistics, result_heap, stats_key):
            passing_measurements = 0
            failing_measurements = 0
            total_configs = 0
            for result in result_heap.results():
                total_configs += 1
                passing_measurements += len(result.passing_measurements())
                failing_measurements += len(result.failing_measurements())

            statistics.set_total_configurations(stats_key, total_configs)
            statistics.set_passing_measurements(stats_key, passing_measurements)
            statistics.set_failing_measurements(stats_key, failing_measurements)

        result_stats = ResultStatistics()
        for model_name, result_heap in self._per_model_sorted_results.items():
            _update_stats(result_stats, result_heap, model_name)

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

    def _create_concurrent_analysis_model_name(self):
        analysis_model_names = [
            model.model_name() for model in self._config.analysis_models
        ]

        self._concurrent_analysis_model_name = ','.join(analysis_model_names)

    def _analyzing_models_concurrently(self):
        """
        Returns
        -------
        bool: True if we are doing concurrent model analysis
        """
        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        return bool(
            results.get_model_measurements_dict(
                models_name=self._concurrent_analysis_model_name,
                suppress_warning=True) and
            len(self._config.analysis_models) > 1)

    def _setup_for_concurrent_analysis(self):
        self._analysis_model_names = [self._concurrent_analysis_model_name]

        model_objectives_list = [
            model.objectives() for model in self._config.analysis_models
        ]
        model_constraints_list = [
            model.constraints() for model in self._config.analysis_models
        ]

        self._run_comparators = {
            self._concurrent_analysis_model_name:
                RunConfigResultComparator(
                    metric_objectives_list=model_objectives_list)
        }

        self._run_constraints = {
            self._concurrent_analysis_model_name: model_constraints_list
        }

    def _setup_for_sequential_analysis(self):
        self._analysis_model_names = [
            model.model_name() for model in self._config.analysis_models
        ]

        self._run_comparators = {
            model.model_name(): RunConfigResultComparator(
                metric_objectives_list=[model.objectives()])
            for model in self._config.analysis_models
        }

        self._run_constraints = {
            model.model_name(): model.constraints()
            for model in self._config.analysis_models
        }

    def _add_results_to_heaps(self):
        """
        Construct and add results to individual result heaps 
        as well as global result heap
        """
        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        for model_name in self._analysis_model_names:
            model_measurements = results.get_model_measurements_dict(model_name)

            if not model_measurements:
                raise TritonModelAnalyzerException(
                    f"The model {model_name} was not found in the loaded checkpoint."
                )

            for (run_config,
                 run_config_measurements) in model_measurements.values():
                run_config_result = RunConfigResult(
                    model_name=model_name,
                    run_config=run_config,
                    comparator=self._run_comparators[model_name],
                    constraints=self._run_constraints[model_name])

                for run_config_measurement in run_config_measurements.values():
                    run_config_measurement.set_metric_weightings(
                        self._run_comparators[model_name]._metric_weights)

                    run_config_measurement.set_model_config_weighting(
                        self._run_comparators[model_name]._model_weights)

                    run_config_result.add_run_config_measurement(
                        run_config_measurement)

                self._per_model_sorted_results[model_name].add_result(
                    run_config_result)
                self._across_model_sorted_results.add_result(run_config_result)

    def _add_default_to_results(self, model_name, results, result_heap):
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

        for run_config_result in result_heap.results():
            if run_config_result.run_config().model_variants_name(
            ) == default_model_name:
                results.append(run_config_result)
                return
