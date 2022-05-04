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

from model_analyzer.constants import LOGGER_NAME
from .model_manager import ModelManager
from .result.result_manager import ResultManager
from .record.metrics_manager import MetricsManager
from .reports.report_manager import ReportManager

from .config.input.config_command_analyze \
    import ConfigCommandAnalyze
from .config.input.config_command_report \
    import ConfigCommandReport
from .config.input.config_command_profile \
    import ConfigCommandProfile
from .config.input.config_defaults import \
    DEFAULT_CHECKPOINT_DIRECTORY
from .model_analyzer_exceptions \
    import TritonModelAnalyzerException

import logging

logger = logging.getLogger(LOGGER_NAME)


class Analyzer:
    """
    A class responsible for coordinating the various components of the
    model_analyzer. Configured with metrics to monitor, exposes profiling and
    result writing methods.
    """

    def __init__(self, config, server, state_manager, checkpoint_required):
        """
        Parameters
        ----------
        config : Config
            Model Analyzer config
        server : TritonServer
            Server handle
        state_manager: AnalyzerStateManager
            The object that maintains Model Analyzer State
        checkpoint_required : bool
            If true, an existing checkpoint is required to run MA
        """

        self._config = config
        self._server = server
        self._state_manager = state_manager
        state_manager.load_checkpoint(checkpoint_required)

        self._result_manager = ResultManager(config=config,
                                             state_manager=self._state_manager)

    def profile(self, client, gpus):
        """
        Subcommand: PROFILE

        Creates a RunConfigGenerator to generate RunConfigs, and then 
        profiles each RunConfig on Perf Analyzer and gathers the resulting
        measurements.
        
        Each RunConfig contains one or more (in the case of concurrent multi-model)
        ModelRunConfigs, each of which contain a ModelConfig and a PerfAnalyzerConfig

        Parameters
        ----------
        client : TritonClient
            Instance used to load/unload models
        gpus: List of GPUDevices
            The gpus being used to profile

        Raises
        ------
        TritonModelAnalyzerException
        """

        if not isinstance(self._config, ConfigCommandProfile):
            raise TritonModelAnalyzerException(
                f"Expected config of type {ConfigCommandProfile},"
                " got {type(self._config)}.")

        self._create_metrics_manager(client, gpus)
        self._create_model_manager(client)
        self._get_server_only_metrics(client)

        self._profile_models()

        logger.info(self._get_profile_complete_string())
        logger.info("")
        logger.info(self._get_analyze_command_help_string())

    def analyze(self, mode, verbose):
        """
        subcommand: ANALYZE

        Constructs results from measurements,
        sorts them, and dumps them to tables.

        Parameters
        ----------
        mode : str
            Global mode that the analyzer is running on
        """

        if not isinstance(self._config, ConfigCommandAnalyze):
            raise TritonModelAnalyzerException(
                f"Expected config of type {ConfigCommandAnalyze}, got {type(self._config)}."
            )

        gpu_info = self._state_manager.get_state_variable('MetricsManager.gpus')
        if not gpu_info:
            gpu_info = {}
        self._report_manager = ReportManager(
            mode=mode,
            config=self._config,
            gpu_info=gpu_info,
            result_manager=self._result_manager)

        # Create result tables, put top results and get stats
        self._result_manager.create_tables()
        self._result_manager.compile_and_sort_results()
        self._report_manager.create_summaries()
        self._report_manager.export_summaries()

        # Dump to tables and write to disk
        self._result_manager.tabulate_results()
        self._result_manager.export_results()
        if verbose:
            self._result_manager.write_results()

        logger.info("")
        logger.info(self._get_report_command_help_string())

    def report(self, mode):
        """
        Subcommand: REPORT

        Generates detailed information on
        one or more model configs

        Parameters
        ----------
        mode : str
            Global mode that the analyzer is running on
        """

        if not isinstance(self._config, ConfigCommandReport):
            raise TritonModelAnalyzerException(
                f"Expected config of type {ConfigCommandReport}, got {type(self._config)}."
            )

        gpu_info = self._state_manager.get_state_variable('MetricsManager.gpus')
        if not gpu_info:
            gpu_info = {}
        self._report_manager = ReportManager(
            mode=mode,
            config=self._config,
            result_manager=self._result_manager,
            gpu_info=gpu_info)

        self._report_manager.create_detailed_reports()
        self._report_manager.export_detailed_reports()

    def _create_metrics_manager(self, client, gpus):
        self._metrics_manager = MetricsManager(
            config=self._config,
            client=client,
            server=self._server,
            gpus=gpus,
            result_manager=self._result_manager,
            state_manager=self._state_manager)

    def _create_model_manager(self, client):
        self._model_manager = ModelManager(
            config=self._config,
            client=client,
            server=self._server,
            result_manager=self._result_manager,
            metrics_manager=self._metrics_manager,
            state_manager=self._state_manager)

    def _get_server_only_metrics(self, client):
        if self._config.triton_launch_mode != 'c_api':
            logger.info('Profiling server only metrics...')
            self._server.start()
            client.wait_for_server_ready(self._config.client_max_retries)
            self._metrics_manager.profile_server()
            self._server.stop()

    def _profile_models(self):

        models = self._config.profile_models

        if self._should_profile_multiple_models_concurrently():
            # Profile all models concurrently
            try:
                self._model_manager.run_models(models=models)
            finally:
                self._state_manager.save_checkpoint()
        else:
            # Profile each model, save state after each
            for model in models:
                if self._state_manager.exiting():
                    break
                try:
                    self._model_manager.run_models(models=[model])
                finally:
                    self._state_manager.save_checkpoint()

    def _should_profile_multiple_models_concurrently(self):
        return (self._config.run_config_profile_models_concurrently_enable and
                len(self._config.profile_models) > 1)

    def _get_profile_complete_string(self):
        profiled_model_list = self._state_manager.get_state_variable(
            'ResultManager.results').get_list_of_models()
        num_profiled_configs = self._get_num_profiled_configs()

        return (f'Profile complete. Profiled {num_profiled_configs} '
                f'configurations for models: {profiled_model_list}')

    def _get_num_profiled_configs(self):
        return sum([
            len(x) for x in self._state_manager.get_state_variable(
                'ResultManager.results').
            get_list_of_model_config_measurement_tuples()
        ])

    def _get_analyze_command_help_string(self):
        return (f'To analyze the profile results and find the best '
                f'configurations, run `{self._get_analyze_command_string()}`')

    def _get_analyze_command_string(self):
        profiled_model_list = self._state_manager.get_state_variable(
            'ResultManager.results').get_list_of_models()

        analyze_command_string = (f'model-analyzer analyze --analysis-models '
                                  f'{",".join(profiled_model_list)}')

        if self._config.config_file is not None:
            analyze_command_string += (f' --config-file '
                                       f'{self._config.config_file}')

        if self._config.checkpoint_directory != DEFAULT_CHECKPOINT_DIRECTORY:
            analyze_command_string += (f' --checkpoint-directory '
                                       f'{self._config.checkpoint_directory}')

        return analyze_command_string

    def _get_report_command_help_string(self):
        top_3_model_config_names = self._get_top_n_model_config_names(n=3)

        return (f'To generate detailed reports for the '
                f'{len(top_3_model_config_names)} best configurations, run '
                f'`{self._get_report_command_string()}`')

    def _get_report_command_string(self):
        top_3_model_config_names = self._get_top_n_model_config_names(n=3)

        report_command_string = (f'model-analyzer report '
                                 f'--report-model-configs '
                                 f'{",".join(top_3_model_config_names)}')

        if self._config.export_path is not None:
            report_command_string += (f' --export-path '
                                      f'{self._config.export_path}')

        if self._config.config_file is not None:
            report_command_string += (f' --config-file '
                                      f'{self._config.config_file}')

        if self._config.checkpoint_directory != DEFAULT_CHECKPOINT_DIRECTORY:
            report_command_string += (f' --checkpoint-directory '
                                      f'{self._config.checkpoint_directory}')

        return report_command_string

    def _get_top_n_model_config_names(self, n=-1):
        return [
            x.run_config().model_variants_name()
            for x in self._result_manager.top_n_results(n=n)
        ]
