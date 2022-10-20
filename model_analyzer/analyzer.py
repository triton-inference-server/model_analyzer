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

from typing import List
import sys
from model_analyzer.constants import LOGGER_NAME
from .model_manager import ModelManager
from .result.result_manager import ResultManager
from .result.result_table_manager import ResultTableManager
from .record.metrics_manager import MetricsManager
from .reports.report_manager import ReportManager
from .config.input.config_command_report \
    import ConfigCommandReport
from .config.input.config_command_profile \
    import ConfigCommandProfile
from .config.input.config_defaults import \
    DEFAULT_CHECKPOINT_DIRECTORY
from .model_analyzer_exceptions \
    import TritonModelAnalyzerException

from .triton.client.client import TritonClient
from .device.gpu_device import GPUDevice

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

    def profile(self, client: TritonClient, gpus: List[GPUDevice], mode: str,
                verbose: bool) -> None:
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
        self._create_model_manager(client, gpus)

        if self._config.model_repository:
            self._get_server_only_metrics(client, gpus)
            self._profile_models()

            # The message is in interrupt_handler(), so we can just exit
            if (self._state_manager.exiting()):
                sys.exit(1)

            logger.info(self._get_profile_complete_string())
            logger.info("")
        elif self._state_manager.starting_fresh_run():
            raise TritonModelAnalyzerException(
                "No model repository specified and no checkpoint found. Please either specify a model repository (-m) or load a checkpoint (--checkpoint-directory)."
            )

        if not self._config.skip_summary_reports:
            self._create_summary_tables(verbose)
            self._create_summary_reports(mode)
            logger.info(self._get_report_command_help_string())

    def report(self, mode: str) -> None:
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

    def _create_model_manager(self, client, gpus):
        self._model_manager = ModelManager(
            config=self._config,
            gpus=gpus,
            client=client,
            server=self._server,
            result_manager=self._result_manager,
            metrics_manager=self._metrics_manager,
            state_manager=self._state_manager)

    def _get_server_only_metrics(self, client, gpus):
        if self._config.triton_launch_mode != 'c_api':
            if not self._state_manager._starting_fresh_run:
                if self._do_checkpoint_gpus_match(gpus):
                    logger.info(
                        "GPU devices match checkpoint - skipping server metric acquisition"
                    )
                    return
                elif gpus is not None:
                    raise TritonModelAnalyzerException(
                        "GPU devices do not match checkpoint - Remove checkpoint file and rerun profile"
                    )

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

    def _create_summary_tables(self, verbose: bool) -> None:
        self._result_table_manager = ResultTableManager(self._config,
                                                        self._result_manager)
        self._result_table_manager.create_tables()
        self._result_table_manager.tabulate_results()
        self._result_table_manager.export_results()

        if verbose:
            self._result_table_manager.write_results()

    def _create_summary_reports(self, mode: str) -> None:
        gpu_info = self._state_manager.get_state_variable('MetricsManager.gpus')
        if not gpu_info:
            gpu_info = {}

        self._report_manager = ReportManager(
            mode=mode,
            config=self._config,
            gpu_info=gpu_info,
            result_manager=self._result_manager)

        self._report_manager.create_summaries()
        self._report_manager.export_summaries()

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

    def _do_checkpoint_gpus_match(self, gpus: dict) -> bool:
        ckpt_data = self._result_manager.get_server_only_data()
        ckpt_uuids = [ckpt_uuid for ckpt_uuid in ckpt_data.keys()]
        gpu_uuids = [gpu._device_uuid for gpu in gpus]

        return sorted(ckpt_uuids) == sorted(gpu_uuids)
